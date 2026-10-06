"""Shared experimental contracts, source lineage, and immutable output helpers."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASPECTS = {
    'design': 'styling and appearance',
    'performance': 'acceleration, handling and driving enjoyment',
    'comfort': 'ride and seating comfort',
    'technology': 'vehicle features and technology',
    'price_value': 'price and value for money',
    'reliability': 'vehicle reliability',
    'maintenance_cost': 'maintenance and repair costs',
    'fuel_economy_range': 'fuel economy and driving range',
    'safety': 'vehicle safety',
}
MODELS = {'nli': 'facebook/bart-large-mnli',
          'tone': 'cardiffnlp/twitter-roberta-base-sentiment-latest'}
DEFAULTS = {
    'version': 'experimental-v1', 'seed': 43, 'threshold': 0.65,
    'margin': 0.15, 'min_opinions': 20, 'min_videos': 5,
    'max_tokens': 512, 'clause_chars': 700,
    'aliases': {'VW': 'VOLKSWAGEN', 'CHEVY': 'CHEVROLET', 'MERCEDES': 'MERCEDES-BENZ'},
}
METHOD = 'pretrained_experimental'
STATUS = 'not_independently_validated'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(text):
    return ' '.join(re.findall(r'\w+', (text or '').casefold()))


def text_hash(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def versions():
    result = {}
    for name in ['polars', 'pyarrow', 'numpy', 'torch', 'transformers', 'huggingface-hub', 'langdetect']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def code_hashes():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def new_dir(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parse_date(value):
    if not value:
        return None
    value = str(value).strip()
    for fmt in ['%m-%d-%Y', '%Y-%m-%d', '%m/%d/%Y']:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date.astimezone(timezone.utc)
    except ValueError:
        return None


def cutoff_date(value):
    date = parse_date(value)
    if date is None:
        raise ValueError('Invalid as-of date')
    # An as-of day includes that entire UTC day.
    return date.replace(hour=23, minute=59, second=59, microsecond=999999)


def make_matcher(makes, aliases):
    terms = {normalized(m): m for m in makes if normalized(m)}
    for alias, make in aliases.items():
        if make in makes:
            terms[normalized(alias)] = make
    # Longest match first prevents short make names within compound makes.
    pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(t) for t in sorted(terms, key=lambda t: (-len(t), t))) + r')(?!\w)')
    def match(text):
        return sorted({terms[m.group(1)] for m in pattern.finditer(normalized(text))})
    return match
