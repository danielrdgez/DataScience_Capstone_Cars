"""Collect YouTube playlist videos and comments into the project SQLite database."""
import argparse
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    from googleapiclient.errors import HttpError
except ImportError:  # pragma: no cover - enables import in test environments without the client library
    class HttpError(Exception):
        def __init__(self, *args: Any, resp: Any = None, content: Any = None, **kwargs: Any):
            super().__init__(*args)
            self.resp = resp
            self.content = content


def _build_youtube_client(api_key: str):
    from googleapiclient.discovery import build

    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


class YouTubeCommentsDatabase:
    """SQLite storage for raw comments and resumable YouTube fetch state."""

    FETCH_STATUS_PENDING = "pending"
    FETCH_STATUS_COMPLETE = "complete"
    FETCH_STATUS_ZERO_COMMENTS = "zero_comments"
    FETCH_STATUS_COMMENTS_DISABLED = "comments_disabled"
    FETCH_STATUS_QUOTA_EXHAUSTED = "quota_exhausted"
    FETCH_STATUS_API_ERROR = "api_error"

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS youtube_comments_sentiment (
                video_id TEXT, playlist_id TEXT, video_title TEXT, source TEXT,
                text TEXT, extracted_at TEXT, comment_id TEXT PRIMARY KEY,
                author TEXT, like_count INTEGER, reply_count INTEGER,
                published_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS youtube_playlist_fetch_state (
                playlist_id TEXT PRIMARY KEY, last_discovered_at TEXT,
                last_status TEXT, last_error TEXT
            );
            CREATE TABLE IF NOT EXISTS youtube_video_fetch_state (
                video_id TEXT PRIMARY KEY, playlist_id TEXT, video_title TEXT,
                discovered_at TEXT, last_attempted_at TEXT, last_succeeded_at TEXT,
                last_status TEXT, last_error TEXT,
                comments_seen_count INTEGER NOT NULL DEFAULT 0,
                next_eligible_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_youtube_video_fetch_status
                ON youtube_video_fetch_state(last_status, next_eligible_at);
            CREATE INDEX IF NOT EXISTS idx_youtube_comments_video
                ON youtube_comments_sentiment(video_id);
        """)
        self.conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def ensure_video_fetch_state(self, video_id: str) -> None:
        now = self._now()
        self.conn.execute("""
            INSERT OR IGNORE INTO youtube_video_fetch_state
                (video_id, discovered_at, last_status, comments_seen_count, next_eligible_at)
            VALUES (?, ?, ?, 0, ?)
        """, (video_id, now, self.FETCH_STATUS_PENDING, now))
        self.conn.commit()

    def get_video_fetch_state(self, video_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM youtube_video_fetch_state WHERE video_id = ?", (video_id,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_playlist_discovery(self, playlist_id: str, videos: List[dict],
                                  status: str = FETCH_STATUS_COMPLETE,
                                  error: Optional[str] = None) -> None:
        now = self._now()
        self.conn.execute("""
            INSERT INTO youtube_playlist_fetch_state VALUES (?, ?, ?, ?)
            ON CONFLICT(playlist_id) DO UPDATE SET
                last_discovered_at=excluded.last_discovered_at,
                last_status=excluded.last_status, last_error=excluded.last_error
        """, (playlist_id, now, status, error))
        for video in videos:
            self.conn.execute("""
                INSERT INTO youtube_video_fetch_state
                    (video_id, playlist_id, video_title, discovered_at,
                     last_status, comments_seen_count, next_eligible_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    playlist_id=COALESCE(excluded.playlist_id, playlist_id),
                    video_title=COALESCE(excluded.video_title, video_title)
            """, (video["video_id"], playlist_id, video.get("title"), now,
                  self.FETCH_STATUS_PENDING, now))
        self.conn.commit()

    def mark_playlist_discovery_error(self, playlist_id: str, status: str, error: str) -> None:
        self.conn.execute("""
            INSERT INTO youtube_playlist_fetch_state VALUES (?, ?, ?, ?)
            ON CONFLICT(playlist_id) DO UPDATE SET
                last_discovered_at=excluded.last_discovered_at,
                last_status=excluded.last_status, last_error=excluded.last_error
        """, (playlist_id, self._now(), status, error))
        self.conn.commit()

    def update_video_fetch_outcome(self, video_id: str, status: str,
                                   comments_seen_count: Optional[int] = None,
                                   error: Optional[str] = None, refresh_days: int = 30,
                                   backoff_hours: int = 6, playlist_id: Optional[str] = None,
                                   video_title: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        succeeded = now.isoformat() if status in {
            self.FETCH_STATUS_COMPLETE, self.FETCH_STATUS_ZERO_COMMENTS,
            self.FETCH_STATUS_COMMENTS_DISABLED,
        } else None
        if status in {self.FETCH_STATUS_COMPLETE, self.FETCH_STATUS_ZERO_COMMENTS,
                      self.FETCH_STATUS_COMMENTS_DISABLED}:
            next_eligible = now + timedelta(days=max(refresh_days, 0))
        else:
            next_eligible = now + timedelta(hours=max(backoff_hours, 0))
        self.conn.execute("""
            INSERT INTO youtube_video_fetch_state
                (video_id, playlist_id, video_title, discovered_at, last_attempted_at,
                 last_succeeded_at, last_status, last_error, comments_seen_count, next_eligible_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                playlist_id=COALESCE(excluded.playlist_id, playlist_id),
                video_title=COALESCE(excluded.video_title, video_title),
                last_attempted_at=excluded.last_attempted_at,
                last_succeeded_at=COALESCE(excluded.last_succeeded_at, last_succeeded_at),
                last_status=excluded.last_status, last_error=excluded.last_error,
                comments_seen_count=COALESCE(excluded.comments_seen_count, comments_seen_count, 0),
                next_eligible_at=excluded.next_eligible_at
        """, (video_id, playlist_id, video_title, now.isoformat(), now.isoformat(),
              succeeded, status, error, comments_seen_count, next_eligible.isoformat()))
        self.conn.commit()

    def insert_sentiment_data(self, rows: List[Dict], table_name: str = "youtube_comments_sentiment") -> int:
        if not rows:
            return 0
        columns = BASE_COLUMNS
        values = [tuple(row.get(column) for column in columns) for row in rows]
        cursor = self.conn.executemany(
            f'INSERT OR IGNORE INTO "{table_name}" ({",".join(columns)}) '
            f'VALUES ({",".join("?" for _ in columns)})', values
        )
        self.conn.commit()
        return max(cursor.rowcount, 0)

    def get_candidate_videos(self, refresh_days: int = 30, force_recheck: bool = False,
                             playlist_ids: Optional[List[str]] = None,
                             video_ids: Optional[List[str]] = None) -> List[dict]:
        now = self._now()
        where = []
        params: List[object] = [self.FETCH_STATUS_PENDING, self.FETCH_STATUS_QUOTA_EXHAUSTED,
                                 self.FETCH_STATUS_API_ERROR, now, now]
        has_comments = "EXISTS (SELECT 1 FROM youtube_comments_sentiment c WHERE c.video_id=s.video_id)"
        eligible = "((s.last_status IS NULL OR s.last_status = ?) AND NOT " + has_comments + ") OR " \
                   "(s.last_status IN (?, ?) AND (s.next_eligible_at IS NULL OR " \
                   "s.next_eligible_at <= ?) AND NOT " + has_comments + ") OR " \
                   "(s.last_status IN ('complete','zero_comments','comments_disabled') " \
                   "AND (s.next_eligible_at IS NULL OR s.next_eligible_at <= ?))"
        if force_recheck:
            eligible = "(" + eligible + " OR s.last_status IS NOT NULL)"
        where.append(eligible)
        if playlist_ids:
            where.append(f"s.playlist_id IN ({','.join('?' for _ in playlist_ids)})")
            params.extend(playlist_ids)
        if video_ids:
            where.append(f"s.video_id IN ({','.join('?' for _ in video_ids)})")
            params.extend(video_ids)
        query = """
            SELECT s.video_id, s.playlist_id, s.video_title
            FROM youtube_video_fetch_state s
            WHERE """ + " AND ".join(where) + " ORDER BY s.discovered_at, s.video_id"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()

BASE_COLUMNS = [
    "video_id",
    "playlist_id",
    "video_title",
    "source",
    "text",
    "extracted_at",
    "comment_id",
    "author",
    "like_count",
    "reply_count",
    "published_at",
    "updated_at",
]

DEFAULT_REFRESH_DAYS = 30
DEFAULT_BACKOFF_HOURS = 6

# Playlists from channels: Auto Buyers Guide, The Car Care Nut, Doug Demuro, Out of Spec Reviews, Throttle House
DEFAULT_PLAYLIST_IDS: List[str] = ['PLdmWqCdsjCu4xf48gB1CvyGXHApkEj8ck', 'PLdmWqCdsjCu6XXTGcfwQE-v-8qZKOUbPv',
                                   'PLdmWqCdsjCu4IAZtt7cFYCI-5wLOu-W5v',
                                   'PLdmWqCdsjCu6vdUJzxywmjNa9IMv9E4bP', 'PLdmWqCdsjCu4Ikxn-hvSiATgUkX5qo2iI',
                                   'PLdmWqCdsjCu6sCNmmkITTmDBGJ1CRJ_ML',
                                   'PLdmWqCdsjCu7ZFuv4M5QZYPHwnK9Da-lG', 'PLdmWqCdsjCu7Y-UCY7jFP1f86lI-IFBES',
                                   'PLdmWqCdsjCu7PO4TJSdIgdIrhAHWGwHvj',
                                   'PLeFzfl0Q8rQXWBaEZdyV1v5u1VIuESa-P', 'PLeFzfl0Q8rQX1NvPXH_Y9OgXqdYNT2TZ-',
                                   'PLeFzfl0Q8rQWfHoQ9mKURykiclndxTn1_',
                                   'PLeFzfl0Q8rQVV48qGZhzQWpapc_F8elah', 'PLeFzfl0Q8rQUy9XXwsTDP44mSCssiNlui',
                                   'PLeFzfl0Q8rQWTHQJQhG_p4yQrseZ9J0wd',
                                   'PLeFzfl0Q8rQW8DEc8TwYyc3LHqwATkHEl', 'PLeFzfl0Q8rQUIskEOInB1h4fUMuNa5P_Y',
                                   'PLeFzfl0Q8rQWY5m2ivI9l6gk_PnkOYA_3',
                                   'PLeFzfl0Q8rQXo-oi6JoPtJNibDi39zj7I', 'PLeFzfl0Q8rQXkSiT50RyVAkC_JIHXw318',
                                   'PLeFzfl0Q8rQUdQRUQJ6qBSn_kwvY3jCDz',
                                   'PLeFzfl0Q8rQVaJp_JfzPZm7oufiqv7yfS', 'PLeFzfl0Q8rQVkkMCL6YhpCO_zjSNhH0Er',
                                   'PLeFzfl0Q8rQXIyh70R-2dKBuSlnFattem',
                                   'PLeFzfl0Q8rQWLwOtDhDNGtdZwgMPAZiQa', 'PLeFzfl0Q8rQXfwL5Nfy3EfNnmxhf6oX1P',
                                   'PLRsapyZqDs1o2_cfJNIxQzb-X1-mpYbdA',
                                   'PLRsapyZqDs1ouavnWh8mpUBZsEjJYSCQI', 'PLRsapyZqDs1ptElTMW3o3PByTZLh4aHmO',
                                   'PLRsapyZqDs1pmgg5rl2ox68CeUm72yG5-',
                                   'PLRsapyZqDs1pC9lvVceSwqUrqx1u8UJ8d', 'PLRsapyZqDs1qzeBRxL7cgjzAY6QEcAaZe',
                                   'PLRsapyZqDs1r0rPWDqedN2NfnSi6BvnTz',
                                   'PLRsapyZqDs1rBVc5yES2M5cJY7sD2nrwN', 'PLRsapyZqDs1rCXGVr98GDWhf6xbHeCjDr',
                                   'PLRsapyZqDs1qq65E6sWXSJELHGVqEeB_0',
                                   'PLRsapyZqDs1psCURs8OOvaxPk-e2WiJZ1', 'PLRsapyZqDs1od8oRXNJvrrINjYPLJ4XUL',
                                   'PLRsapyZqDs1oWFpAszIPLBAggO3i7MynT',
                                   'PLRsapyZqDs1oRf6s2gk6qF0BN_kGKXSKZ', 'PLRsapyZqDs1paxezbET0LJ00M1IHesv0u',
                                   'PLRsapyZqDs1oS4H-h-7PMeNF3bC1Df60q',
                                   'PLRsapyZqDs1oMzw7_bLfVPpHQgefNsxvC', 'PLRsapyZqDs1qjJB9zPXM5gFVCZ3NDUWLH',
                                   'PLRsapyZqDs1oitmbWCluObhfUGQ3ZuvzC',
                                   'PLRsapyZqDs1ryftnOF61p1ez4K576556G', 'PLRsapyZqDs1r3sjTDWALSNHFuzejXk9vf',
                                   'PLRsapyZqDs1rWoLJB2miI4tx_Q4_nAOXe',
                                   'PLVa4b_Vn4gbBWaieOY6Z_zd37XlbHvsG6', 'PLVa4b_Vn4gbDD5IrxxVnYU7GVr_xjp9Zx',
                                   'PLVa4b_Vn4gbCcL-FHtFY9837w0Hw5mAiG',
                                   'PLuLtF2Rwd40U_7Tiia8CFUr62nXVrfQN-', 'PLM8vs0dTZFTJ46frL-sD8u3_yT-PWNJSw',
                                   'PL2ir4svMoaYhRAuKykEGRgsrCk__OJRz3',
                                   'PLJLJaYBuTkgut9r9dQS8Prk1SxKs-uaFu', 'PLsSQoIGhBLpT2eTLmGYwTYGAIKBOJjHnX',
                                   'PLsSQoIGhBLpQwAPiW1fCPh4LEWxlbp9Cx',
                                   'PLsSQoIGhBLpSW5q7K9sKS8j9VnXc3VWVw', 'PLsSQoIGhBLpTYYZtQRbxl2NjIl-gRhXCN']
DEFAULT_VIDEO_IDS: List[str] = []


class QuotaExceededError(RuntimeError):
    pass


def retry_with_exponential_backoff(max_retries: int = 5, base_delay: float = 1.0):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except HttpError as exc:
                    if is_quota_error(exc):
                        delay = base_delay * (2 ** i)
                        logging.warning(
                            "Quota exceeded for %s. Retrying in %.2f seconds... (Attempt %s/%s)",
                            func.__name__,
                            delay,
                            i + 1,
                            max_retries,
                        )
                        time.sleep(delay)
                        continue
                    raise
                except Exception as exc:
                    logging.error("Unexpected error in %s: %s", func.__name__, exc)
                    raise
            raise QuotaExceededError(f"Max retries exceeded for {func.__name__} due to quota exhaustion.")

        return wrapper

    return decorator


def _read_key_from_env_file(env_path: Path) -> Optional[str]:
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"YOUTUBE_API_KEY", "GOOGLE_API_KEY"}:
                return value.strip().strip("\"'") or None
    except FileNotFoundError:
        return None
    return None


def load_api_key(api_key_file: Optional[str] = None) -> Optional[str]:
    key = os.getenv("YOUTUBE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        logging.info("API key loaded from environment variable.")
        return key.strip()

    env_path = Path(__file__).resolve().parent.parent / ".env"
    env_key = _read_key_from_env_file(env_path)
    if env_key:
        logging.info("API key loaded from .env file at: %s", env_path)
        return env_key

    if api_key_file:
        logging.info("Attempting to load API key from file: %s", api_key_file)
        try:
            return Path(api_key_file).read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            logging.error("API key file not found: %s", api_key_file)
    return None


def build_dataframe(rows: Iterable[Dict]) -> List[Dict]:
    return [{column: row.get(column) for column in BASE_COLUMNS} for row in rows]


def _decode_http_content(exc: HttpError) -> str:
    try:
        if isinstance(exc.content, bytes):
            return exc.content.decode("utf-8", errors="ignore")
        return str(exc.content or "")
    except Exception:
        return ""


def _extract_error_reason(exc: HttpError) -> str:
    payload = _decode_http_content(exc)
    if not payload:
        return str(exc)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload
    errors = parsed.get("error", {}).get("errors", [])
    if errors:
        reason = errors[0].get("reason")
        message = errors[0].get("message")
        if reason and message:
            return f"{reason}: {message}"
        if reason:
            return str(reason)
        if message:
            return str(message)
    return parsed.get("error", {}).get("message", payload)


def is_quota_error(exc: HttpError | Exception) -> bool:
    if isinstance(exc, QuotaExceededError):
        return True
    if not isinstance(exc, HttpError):
        return False
    if getattr(exc, "resp", None) is None:
        return True
    if exc.resp.status != 403:
        return False
    reason = _extract_error_reason(exc).lower()
    return "quota" in reason or "daily limit" in reason or "rate limit" in reason


def is_comments_disabled_error(exc: HttpError) -> bool:
    if not isinstance(exc, HttpError):
        return False
    reason = _extract_error_reason(exc).lower()
    return "commentsdisabled" in reason or "comments disabled" in reason


@retry_with_exponential_backoff()
def fetch_comments(
    video_id: str,
    api_key: str,
    max_comments: int,
    order: str,
    video_title: Optional[str] = None,
    playlist_id: Optional[str] = None,
) -> List[Dict]:
    youtube = _build_youtube_client(api_key)
    comments: List[Dict] = []
    remaining = max_comments if max_comments > 0 else -1
    page_token = None

    while remaining != 0:
        batch_size = 100 if remaining < 0 else min(100, remaining)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=batch_size,
            order=order,
            textFormat="plainText",
            pageToken=page_token,
        )
        response = request.execute()
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            top_comment = snippet.get("topLevelComment", {}).get("snippet", {})
            comments.append(
                {
                    "video_id": video_id,
                    "playlist_id": playlist_id,
                    "video_title": video_title,
                    "source": "comment",
                    "text": top_comment.get("textOriginal") or top_comment.get("textDisplay") or "",
                    "comment_id": item.get("id"),
                    "author": top_comment.get("authorDisplayName"),
                    "like_count": top_comment.get("likeCount"),
                    "reply_count": snippet.get("totalReplyCount"),
                    "published_at": top_comment.get("publishedAt"),
                    "updated_at": top_comment.get("updatedAt"),
                }
            )
            if remaining > 0:
                remaining -= 1
                if remaining == 0:
                    break
        if remaining == 0:
            break
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return comments


@retry_with_exponential_backoff()
def fetch_video_title(video_id: str, api_key: str) -> Optional[str]:
    youtube = _build_youtube_client(api_key)
    request = youtube.videos().list(part="snippet", id=video_id, maxResults=1)
    response = request.execute()
    items = response.get("items", [])
    if not items:
        return None
    return items[0].get("snippet", {}).get("title")


@retry_with_exponential_backoff()
def fetch_playlist_videos(playlist_id: str, api_key: str, max_videos: int) -> List[Dict]:
    youtube = _build_youtube_client(api_key)
    videos: List[Dict] = []
    remaining = max_videos if max_videos > 0 else -1
    page_token = None

    while remaining != 0:
        batch_size = 50 if remaining < 0 else min(50, remaining)
        request = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=batch_size,
            pageToken=page_token,
        )
        response = request.execute()
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            resource = snippet.get("resourceId", {})
            video_id = resource.get("videoId")
            if not video_id:
                continue
            videos.append({"video_id": video_id, "title": snippet.get("title")})
            if remaining > 0:
                remaining -= 1
                if remaining == 0:
                    break
        if remaining == 0:
            break
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return videos


def discover_playlist_videos(
    db: YouTubeCommentsDatabase,
    playlist_id: str,
    api_key: str,
    max_videos: int,
) -> List[Dict]:
    try:
        videos = fetch_playlist_videos(playlist_id, api_key, max_videos)
        if not videos:
            db.upsert_playlist_discovery(playlist_id, [], status=YouTubeCommentsDatabase.FETCH_STATUS_COMPLETE)
            logging.warning("No videos found for playlist %s.", playlist_id)
            return []
        db.upsert_playlist_discovery(playlist_id, videos, status=YouTubeCommentsDatabase.FETCH_STATUS_COMPLETE)
        return videos
    except (HttpError, QuotaExceededError) as exc:
        status = (
            YouTubeCommentsDatabase.FETCH_STATUS_QUOTA_EXHAUSTED
            if is_quota_error(exc)
            else YouTubeCommentsDatabase.FETCH_STATUS_API_ERROR
        )
        db.mark_playlist_discovery_error(playlist_id, status, str(_extract_error_reason(exc) if isinstance(exc, HttpError) else exc))
        raise


def process_video(
    db: YouTubeCommentsDatabase,
    video_id: str,
    api_key: str,
    max_comments: int,
    order: str,
    refresh_days: int,
    playlist_id: Optional[str] = None,
    video_title: Optional[str] = None,
) -> dict:
    state = db.get_video_fetch_state(video_id) or {}
    playlist_id = playlist_id or state.get("playlist_id")
    video_title = video_title or state.get("video_title")

    if not video_title:
        try:
            video_title = fetch_video_title(video_id, api_key)
        except (HttpError, QuotaExceededError) as exc:
            error_text = _extract_error_reason(exc) if isinstance(exc, HttpError) else str(exc)
            if is_quota_error(exc):
                db.update_video_fetch_outcome(
                    video_id=video_id,
                    playlist_id=playlist_id,
                    video_title=video_title,
                    status=YouTubeCommentsDatabase.FETCH_STATUS_QUOTA_EXHAUSTED,
                    error=error_text,
                    refresh_days=refresh_days,
                    backoff_hours=DEFAULT_BACKOFF_HOURS,
                )
                raise QuotaExceededError(error_text) from exc
            logging.warning("Unable to fetch title for %s: %s", video_id, error_text)

    extracted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        rows = fetch_comments(
            video_id=video_id,
            api_key=api_key,
            max_comments=max_comments,
            order=order,
            video_title=video_title,
            playlist_id=playlist_id,
        )
    except (HttpError, QuotaExceededError) as exc:
        if isinstance(exc, HttpError) and is_comments_disabled_error(exc):
            db.update_video_fetch_outcome(
                video_id=video_id,
                playlist_id=playlist_id,
                video_title=video_title,
                status=YouTubeCommentsDatabase.FETCH_STATUS_COMMENTS_DISABLED,
                comments_seen_count=0,
                error=_extract_error_reason(exc),
                refresh_days=refresh_days,
                backoff_hours=DEFAULT_BACKOFF_HOURS,
            )
            return {"status": YouTubeCommentsDatabase.FETCH_STATUS_COMMENTS_DISABLED, "inserted_count": 0}

        error_text = _extract_error_reason(exc) if isinstance(exc, HttpError) else str(exc)
        status = (
            YouTubeCommentsDatabase.FETCH_STATUS_QUOTA_EXHAUSTED
            if is_quota_error(exc)
            else YouTubeCommentsDatabase.FETCH_STATUS_API_ERROR
        )
        db.update_video_fetch_outcome(
            video_id=video_id,
            playlist_id=playlist_id,
            video_title=video_title,
            status=status,
            comments_seen_count=0,
            error=error_text,
            refresh_days=refresh_days,
            backoff_hours=DEFAULT_BACKOFF_HOURS,
        )
        if status == YouTubeCommentsDatabase.FETCH_STATUS_QUOTA_EXHAUSTED:
            raise QuotaExceededError(error_text) from exc
        return {"status": status, "inserted_count": 0}

    for row in rows:
        row["extracted_at"] = extracted_at

    df = build_dataframe(rows)
    inserted_count = db.insert_sentiment_data(df, table_name="youtube_comments_sentiment")
    status = (
        YouTubeCommentsDatabase.FETCH_STATUS_ZERO_COMMENTS
        if not df
        else YouTubeCommentsDatabase.FETCH_STATUS_COMPLETE
    )
    db.update_video_fetch_outcome(
        video_id=video_id,
        playlist_id=playlist_id,
        video_title=video_title,
        status=status,
        comments_seen_count=len(df),
        error=None,
        refresh_days=refresh_days,
        backoff_hours=DEFAULT_BACKOFF_HOURS,
    )
    logging.info(
        "Processed video %s (%s): fetched=%s inserted=%s status=%s",
        video_id,
        playlist_id or "no-playlist",
        len(df),
        inserted_count,
        status,
    )
    return {"status": status, "inserted_count": inserted_count, "fetched_count": len(df)}


def discover_default_targets(
    db: YouTubeCommentsDatabase,
    api_key: str,
    max_videos: int,
) -> None:
    for playlist_id in DEFAULT_PLAYLIST_IDS:
        try:
            discover_playlist_videos(db, playlist_id, api_key, max_videos)
        except QuotaExceededError:
            raise
        except HttpError as exc:
            logging.warning("Skipping playlist discovery for %s due to API error: %s", playlist_id, _extract_error_reason(exc))

    for video_id in DEFAULT_VIDEO_IDS:
        db.ensure_video_fetch_state(video_id)


def run_queue(
    db: YouTubeCommentsDatabase,
    api_key: str,
    max_comments: int,
    order: str,
    refresh_days: int,
    force_recheck: bool,
    stop_on_quota: bool,
    playlist_ids: Optional[List[str]] = None,
    video_ids: Optional[List[str]] = None,
) -> dict:
    candidates = db.get_candidate_videos(
        refresh_days=refresh_days,
        force_recheck=force_recheck,
        playlist_ids=playlist_ids,
        video_ids=video_ids,
    )
    logging.info("Queue contains %s eligible videos.", len(candidates))

    summary = {
        "processed_videos": 0,
        "inserted_comments": 0,
        "stopped_on_quota": False,
    }
    for candidate in candidates:
        try:
            result = process_video(
                db=db,
                video_id=candidate["video_id"],
                playlist_id=candidate.get("playlist_id"),
                video_title=candidate.get("video_title"),
                api_key=api_key,
                max_comments=max_comments,
                order=order,
                refresh_days=refresh_days,
            )
            summary["processed_videos"] += 1
            summary["inserted_comments"] += int(result.get("inserted_count", 0))
        except QuotaExceededError as exc:
            logging.warning("Quota exhaustion while processing %s: %s", candidate["video_id"], exc)
            summary["stopped_on_quota"] = True
            if stop_on_quota:
                break
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch YouTube comments with the official API and save comment data and fetch state to SQLite."
    )
    parser.add_argument("--video-id", help="YouTube video id")
    parser.add_argument("--playlist-id", help="YouTube playlist id")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Max videos to fetch from playlist discovery (0 for all)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=0,
        help="Max comments to fetch per video (0 for all)",
    )
    parser.add_argument(
        "--comment-order",
        choices=["relevance", "time"],
        default="relevance",
        help="Ordering for comments",
    )
    parser.add_argument("--api-key-file", help="Path to API key file")
    parser.add_argument(
        "--output-db",
        default=os.path.join(Path(__file__).resolve().parent.parent, "DATA", "CAR_DATA.db"),
        help="Output SQLite database path (default: DATA/CAR_DATA.db)",
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=DEFAULT_REFRESH_DAYS,
        help="Days to wait before rechecking completed or zero-comment videos.",
    )
    parser.add_argument(
        "--force-recheck",
        action="store_true",
        help="Include fresh completed videos after higher-priority unseen and retryable videos.",
    )
    parser.add_argument(
        "--stop-on-quota",
        dest="stop_on_quota",
        action="store_true",
        default=True,
        help="Stop the run cleanly when the YouTube quota is exhausted (default: true).",
    )
    parser.add_argument(
        "--continue-on-quota",
        dest="stop_on_quota",
        action="store_false",
        help="Continue scanning the queue after quota-like errors instead of stopping immediately.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    api_key = load_api_key(args.api_key_file)
    if not api_key:
        raise ValueError("Missing API key. Set YOUTUBE_API_KEY/GOOGLE_API_KEY or use --api-key-file.")

    db_path = Path(args.output_db)
    db = YouTubeCommentsDatabase(str(db_path))
    try:
        if args.video_id:
            db.ensure_video_fetch_state(args.video_id)
            summary = run_queue(
                db=db,
                api_key=api_key,
                max_comments=args.max_comments,
                order=args.comment_order,
                refresh_days=args.refresh_days,
                force_recheck=True,
                stop_on_quota=args.stop_on_quota,
                video_ids=[args.video_id],
            )
        elif args.playlist_id:
            discover_playlist_videos(db, args.playlist_id, api_key, args.max_videos)
            summary = run_queue(
                db=db,
                api_key=api_key,
                max_comments=args.max_comments,
                order=args.comment_order,
                refresh_days=args.refresh_days,
                force_recheck=args.force_recheck,
                stop_on_quota=args.stop_on_quota,
                playlist_ids=[args.playlist_id],
            )
        else:
            discover_default_targets(db, api_key, args.max_videos)
            summary = run_queue(
                db=db,
                api_key=api_key,
                max_comments=args.max_comments,
                order=args.comment_order,
                refresh_days=args.refresh_days,
                force_recheck=args.force_recheck,
                stop_on_quota=args.stop_on_quota,
            )

        logging.info(
            "Finished comment ingestion. processed_videos=%s inserted_comments=%s stopped_on_quota=%s",
            summary["processed_videos"],
            summary["inserted_comments"],
            summary["stopped_on_quota"],
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
