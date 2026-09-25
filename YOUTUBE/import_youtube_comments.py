"""Copy youtube_comments_sentiment into DATA/CAR_DATA.db in batches."""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = Path(r"E:\Car-Price-Data-Visualization-Learning\CAR_DATA_OUTPUT\CAR_DATA_FINAL.db")
DEST_DB = ROOT / "DATA" / "CAR_DATA.db"
TABLE = "youtube_comments_sentiment"
COLUMNS = ("video_id", "playlist_id", "video_title", "source", "text",
           "extracted_at", "comment_id", "author", "like_count", "reply_count",
           "published_at", "updated_at")


def import_comments(source: Path, destination: Path, batch_size: int = 5000) -> int:
    if not source.exists():
        raise FileNotFoundError(f"YouTube source database not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{source.resolve().as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(destination, timeout=60)
    total = 0
    try:
        src.execute("PRAGMA query_only=ON")
        available = {row[1] for row in src.execute(f'PRAGMA table_info("{TABLE}")')}
        missing = set(COLUMNS) - available
        if missing:
            raise ValueError(f"Source {TABLE} is missing columns: {sorted(missing)}")
        dst.execute(f'''CREATE TABLE IF NOT EXISTS {TABLE} (
            video_id TEXT, playlist_id TEXT, video_title TEXT, source TEXT,
            text TEXT, extracted_at DATE, comment_id TEXT PRIMARY KEY, author TEXT,
            like_count INTEGER, reply_count INTEGER, published_at DATE, updated_at DATE
        )''')
        fields = ",".join(f'"{c}"' for c in COLUMNS)
        marks = ",".join("?" for _ in COLUMNS)
        sql = f'INSERT OR REPLACE INTO "{TABLE}" ({fields}) VALUES ({marks})'
        cursor = src.execute(f'SELECT {fields} FROM "{TABLE}"')
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            dst.executemany(sql, batch)
            dst.commit()
            total += len(batch)
        dst.execute('CREATE INDEX IF NOT EXISTS idx_youtube_video_id ON youtube_comments_sentiment(video_id)')
        dst.commit()
        return total
    finally:
        dst.close()
        src.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DB)
    parser.add_argument("--db", type=Path, default=DEST_DB)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    print(f"Imported {import_comments(args.source, args.db, args.batch_size):,} YouTube comments into {args.db}")


if __name__ == "__main__":
    main()
