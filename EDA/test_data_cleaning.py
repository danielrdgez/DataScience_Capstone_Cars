"""Checks for listing transformations, source preservation, and Parquet export.

Run from the project root with: python -m unittest discover -s EDA -p "test_*.py"
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import polars as pl

from DATA_CLEANING import (
    clean_used_cars_batch, clean_vpic_batch, load_data, export_parquet,
    load_final_data, scan_final_table, USED_CARS_FLOAT_COLUMNS,
    USED_CARS_INTEGER_COLUMNS, USED_CARS_UNITS, TABLES_TO_EXPORT,
    INTEGER_COLUMNS, FLOAT_COLUMNS, PARQUET_FILES,
)


class SourcePreservationTests(unittest.TestCase):
    def test_listing_values_follow_requested_types(self):
        source = pl.DataFrame({
            "owner_count": ["3.0", "2.5", "", None],
            "dealer_zip": ["00960", "00922", "00100", None],
            "height": ["35.1 in", "--", " 42 in ", None],
            "price": ["1.0", "invalid", "", None],
            "listed_date": ["2020-01-01", "invalid", "", None],
        })
        result = clean_used_cars_batch(source)
        self.assertEqual(result['owner_count'].to_list(), [3, 3, None, None])
        self.assertEqual(result['dealer_zip'].to_list(), [960, 922, 100, None])
        self.assertEqual(result['height'].to_list(), [35.1, None, 42.0, None])
        self.assertEqual(result['price'].to_list(), [1, None, None, None])
        self.assertEqual(str(result['listed_date'][0]), '2020-01-01')
        self.assertEqual(result['listed_date'].null_count(), 3)

    def test_vpic_values_survive_column_renaming(self):
        source = pl.DataFrame({
            "vpic_Doors": ["4", "4.0", "", None],
            "vpic_BasePrice": ["1.0", "--", "", None],
            "vpic_ABS": ["Standard", "Not Applicable", "", None],
            "fetched_at": ["2026-09-25 20:23:26", "invalid", "", None],
        })
        result = clean_vpic_batch(source)
        self.assertEqual(result['Doors'].to_list(), [4, 4, None, None])
        self.assertEqual(result['BasePrice'].to_list(), [1.0, None, None, None])
        self.assertEqual(result['ABS'].to_list(), source['vpic_ABS'].to_list())
        self.assertEqual(result['fetched_at'].to_list(), source['fetched_at'].to_list())

    def test_valid_types_change_without_new_nulls(self):
        source = pl.DataFrame({
            "vpic_Doors": ["4", None],
            "vpic_BasePrice": ["1.5", None],
            "fetched_at": ["2026-09-25 20:23:26", None],
        })
        result = clean_vpic_batch(source)
        self.assertEqual(result["Doors"].dtype, pl.Int64)
        self.assertEqual(result["BasePrice"].dtype, pl.Float64)
        self.assertEqual(result["fetched_at"].dtype, pl.Datetime("us"))
        self.assertEqual(result["Doors"].to_list(), [4, None])
        self.assertEqual(result["BasePrice"].to_list(), [1.5, None])
        self.assertEqual(result.null_count().row(0), (1, 1, 1))
        self.assertEqual(result.height, source.height)

    def test_all_requested_fields_units_and_rounding(self):
        source = {}
        for name in USED_CARS_FLOAT_COLUMNS:
            unit = USED_CARS_UNITS.get(name, '')
            source[name] = [f' 35.1 {unit} '.strip(), '', None, '--']
        for name in USED_CARS_INTEGER_COLUMNS:
            source[name] = ['2.5', '-2.5', '2.49999999999999999999', '3.0']
        source['combine_fuel_economy'] = ['1.0', '', None, '--']
        source['maximum_seating'] = ['5 seats', '', None, '--']
        result = clean_used_cars_batch(pl.DataFrame(source))
        for name in USED_CARS_FLOAT_COLUMNS:
            self.assertEqual(result[name].dtype, pl.Float64)
            self.assertEqual(result[name].to_list(), [35.1, None, None, None])
        for name in USED_CARS_INTEGER_COLUMNS:
            self.assertEqual(result[name].dtype, pl.Int64)
            self.assertEqual(result[name].to_list(), [3, -3, 2, 3])
        for name in ('combine_fuel_economy', 'maximum_seating'):
            self.assertEqual(result[name].to_list(), source[name])

    def test_overflow_invalids_and_exact_large_integers(self):
        source = pl.DataFrame({
            'price': ['9223372036854775807.0', '9223372036854775808',
                      '1e500', 'invalid', '-9223372036854775808.0'],
            'height': ['inf', 'NaN', '1e500', '12 cm', None],
        })
        result = clean_used_cars_batch(source)
        self.assertEqual(result['price'].to_list(), [9223372036854775807, None, None, None, -9223372036854775808])
        self.assertEqual(result['height'].to_list(), [None] * 5)

    def test_loader_preserves_rows_json_and_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "source.db"
            with sqlite3.connect(database) as connection:
                connection.execute('CREATE TABLE used_cars (dealer_zip TEXT, height TEXT)')
                connection.executemany('INSERT INTO used_cars VALUES (?, ?)',
                                       [("00960", "35.1 in"), ("", "--"), (None, None)])
                connection.execute('CREATE TABLE nhtsa_complaints (record_json TEXT)')
                record = '{"summary": "", "vin": null, "crash": false, "numberOfDeaths": 0}'
                connection.execute('INSERT INTO nhtsa_complaints VALUES (?)', (record,))
            connection.close()
            before = database.read_bytes()
            rows = []
            for table, batch in load_data(database, batch_size=2,
                                          table_names=("used_cars", "nhtsa_complaints")):
                if table == "used_cars":
                    rows.extend(batch.rows())
                else:
                    self.assertEqual(batch["record_json"].to_list(), [record])
            self.assertEqual(rows, [(960, 35.1), (None, None), (None, None)])
            self.assertEqual(database.read_bytes(), before)


class ParquetExportTests(unittest.TestCase):
    def test_round_trip_fixed_schema_and_default_scope(self):
        self.assertEqual(len(TABLES_TO_EXPORT), 5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / 'source.db'
            with sqlite3.connect(database) as connection:
                connection.execute('CREATE TABLE used_cars (vin TEXT, price TEXT, height TEXT, listed_date TEXT, dealer_zip TEXT)')
                connection.executemany('INSERT INTO used_cars VALUES (?,?,?,?,?)', [
                    ('a', '', None, None, ''), ('b', '12.5', '66.5 in', '2020-01-01', '00960'),
                    ('c', '12.49', '--', 'invalid', None),
                ])
                connection.execute('CREATE TABLE nhtsa_vpic_decodes (vpic_Doors TEXT, vpic_BasePrice TEXT, fetched_at TEXT, vpic_ABS TEXT)')
                connection.executemany('INSERT INTO nhtsa_vpic_decodes VALUES (?,?,?,?)', [
                    ('04', '1.5', '2026-09-25 20:23:26', None),
                    ('4.0', '2.0', '2026-09-25 20:23:27', 'Standard'),
                ])
                connection.execute('CREATE TABLE nhtsa_recalls (query_id INTEGER, record_json TEXT)')
            connection.close()
            before = database.read_bytes()
            output = root / 'final'
            tables = ('used_cars', 'nhtsa_vpic_decodes', 'nhtsa_recalls')
            report = export_parquet(database, output, batch_size=1, table_names=tables)
            self.assertEqual(database.read_bytes(), before)
            self.assertEqual(report['tables']['used_cars']['rows'], 3)
            self.assertEqual(report['tables']['used_cars']['new_nulls_from_conversion']['price'], 1)
            self.assertEqual(report['tables']['used_cars']['file'], 'used_cars.parquet')
            actual = scan_final_table(dataset_path=output).collect()
            self.assertEqual(actual['price'].to_list(), [None, 13, 12])
            self.assertEqual(actual['price'].dtype, pl.Int64)
            self.assertEqual(actual['height'].to_list(), [None, 66.5, None])
            self.assertEqual(actual['listed_date'].dtype, pl.Date)
            vpic = scan_final_table('nhtsa_vpic_decodes', output).collect()
            # Fixed numeric dtypes are the same in every saved batch.
            self.assertEqual(vpic['Doors'].to_list(), [4, 4])
            self.assertEqual(vpic['BasePrice'].dtype, pl.Float64)
            self.assertEqual(vpic['ABS'].to_list(), [None, 'Standard'])
            empty = scan_final_table('nhtsa_recalls', output).collect()
            self.assertEqual(empty.schema, {'query_id': pl.Int64, 'record_json': pl.String})
            batches = list(load_final_data(output, batch_size=2, table_names=("used_cars",)))
            self.assertEqual([name for name, _ in batches], ['used_cars', 'used_cars'])
            self.assertTrue(pl.concat([batch for _, batch in batches]).equals(actual))
            with self.assertRaises(FileExistsError):
                export_parquet(database, output)
            with self.assertRaises(RuntimeError):
                export_parquet(database, root / 'missing', table_names=('nhtsa_complaints',))
            self.assertFalse((root / 'missing').exists())
            self.assertFalse(list(root.glob('.car-data-export-*')))

    def test_empty_listing_file_has_requested_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / 'source.db'
            with sqlite3.connect(database) as connection:
                connection.execute('CREATE TABLE used_cars (price TEXT, listed_date TEXT)')
            connection.close()
            export_parquet(database, root / 'final', table_names=('used_cars',))
            self.assertEqual(scan_final_table(dataset_path=root / 'final').collect_schema(),
                             {'price': pl.Int64, 'listed_date': pl.Date})


class NHTSATransformationTests(unittest.TestCase):
    def test_all_vpic_numeric_fields(self):
        source = {('model_year_hint' if c == 'model_year_hint' else 'vpic_' + c):
                  ['2.5', '-2.5', 'bad', None] for c in INTEGER_COLUMNS}
        source.update({'vpic_' + c: ['35.25', '1e500', '', None] for c in FLOAT_COLUMNS})
        source['vpic_ABS'] = ['Standard', 'Optional', '', None]
        result = clean_vpic_batch(pl.DataFrame(source))
        for c in INTEGER_COLUMNS:
            self.assertEqual(result[c].dtype, pl.Int64)
            self.assertEqual(result[c].to_list(), [3, -3, None, None])
        for c in FLOAT_COLUMNS:
            self.assertEqual(result[c].dtype, pl.Float64)
            self.assertEqual(result[c].to_list(), [35.25, None, None, None])
        self.assertFalse(any(c.startswith('vpic_') for c in result.columns))
        self.assertEqual(result['ABS'].to_list(), source['vpic_ABS'])

    def test_json_union_collisions_and_safety_pivot(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / 'source.db'
            records = [
                {'summary': None, 'crash': False, 'numberOfDeaths': 0, 'products': [{'productYear': '2020'}],
                 'make': 'source make', 'json_make': 'already named', 'mixed': 2},
                {'summary': '', 'crash': True, 'numberOfDeaths': 2, 'products': [],
                 'late_key': 'appeared later', 'mixed': 'two'},
            ]
            with sqlite3.connect(database) as connection:
                connection.execute('CREATE TABLE used_cars (price TEXT)')
                connection.execute('CREATE TABLE nhtsa_vpic_decodes (vpic_Doors TEXT)')
                for table in ('nhtsa_complaints', 'nhtsa_recalls'):
                    connection.execute(f'CREATE TABLE {table} (query_id INTEGER, model_year TEXT, make TEXT, record_json TEXT)')
                    connection.executemany(f'INSERT INTO {table} VALUES (?,?,?,?)',
                        [(1, '2020.0', 'listing make', json.dumps(records[0])),
                         (2, '2021.5', 'other make', json.dumps(records[1]))])
                connection.execute('CREATE TABLE nhtsa_safety_rating_values (query_id INTEGER, vehicle_id TEXT, field_name TEXT, field_value TEXT)')
                connection.executemany('INSERT INTO nhtsa_safety_rating_values VALUES (?,?,?,?)', [
                    (1, 'a', 'OverallRating', '5'), (1, 'a', 'dynamicTipResult', ' '),
                    (1, 'b', 'OverallRating', 'Not Rated'), (2, 'a', 'OverallRating', '4'),
                    (1, 'b', 'SideRating', None), (2, 'a', 'SideRating', ''),
                ])
            connection.close()
            before = database.read_bytes()
            tables = ('nhtsa_complaints', 'nhtsa_recalls', 'nhtsa_safety_rating_values')
            report = export_parquet(database, root / 'final', batch_size=1)
            self.assertEqual(set(report['tables']), set(TABLES_TO_EXPORT))
            self.assertEqual({path.name for path in (root / 'final').glob('*.parquet')}, set(PARQUET_FILES.values()))
            self.assertEqual(database.read_bytes(), before)
            for table in tables[:2]:
                result = scan_final_table(table, root / 'final').collect()
                self.assertEqual(result['model_year'].to_list(), [2020, 2022])
                self.assertEqual(result['late_key'].to_list(), [None, 'appeared later'])
                self.assertEqual(result['summary'].to_list(), [None, ''])
                self.assertEqual(result['crash'].to_list(), [False, True])
                self.assertEqual(result['numberOfDeaths'].dtype, pl.Int64)
                self.assertEqual(result['mixed'].to_list(), ['2', 'two'])
                self.assertEqual(result['make'].to_list(), ['listing make', 'other make'])
                self.assertEqual(result['json_json_make'].to_list(), ['source make', None])
                self.assertEqual(json.loads(result['products'][0]), records[0]['products'])
                self.assertEqual(json.loads(result['record_json'][1]), records[1])
                self.assertTrue((root / 'final' / PARQUET_FILES[table]).is_file())
            safety = scan_final_table('nhtsa_safety_rating_values', root / 'final').collect()
            self.assertEqual(safety.height, 3)
            self.assertEqual(safety.select('query_id', 'vehicle_id').rows(), [(1, 'a'), (1, 'b'), (2, 'a')])
            self.assertEqual(safety['OverallRating'].to_list(), ['5', 'Not Rated', '4'])
            self.assertEqual(safety['dynamicTipResult'].to_list(), [' ', None, None])
            self.assertEqual(safety['SideRating'].to_list(), [None, None, ''])
            self.assertNotIn('field_name', safety.columns)
            self.assertNotIn('field_value', safety.columns)
            self.assertEqual(report['tables']['nhtsa_safety_rating_values']['source_rows'], 6)
            self.assertEqual(report['tables']['nhtsa_safety_rating_values']['rows'], 3)

    def test_invalid_json_and_duplicate_safety_abort_without_publication(self):
        for raw in ('not json', '[]', '{"key":1,"key":2}'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                database = root / 'source.db'
                with sqlite3.connect(database) as connection:
                    connection.execute('CREATE TABLE nhtsa_complaints (record_json TEXT)')
                    connection.execute('INSERT INTO nhtsa_complaints VALUES (?)', (raw,))
                connection.close()
                with self.assertRaisesRegex(ValueError, 'Invalid record_json'):
                    export_parquet(database, root / 'final', table_names=('nhtsa_complaints',))
                self.assertFalse((root / 'final').exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / 'source.db'
            with sqlite3.connect(database) as connection:
                connection.execute('CREATE TABLE nhtsa_safety_rating_values (query_id INTEGER, vehicle_id TEXT, field_name TEXT, field_value TEXT)')
                connection.executemany('INSERT INTO nhtsa_safety_rating_values VALUES (1, "a", "Rating", ?)', [('5',), ('4',)])
            connection.close()
            with self.assertRaisesRegex(ValueError, 'Duplicate safety field'):
                export_parquet(database, root / 'final', table_names=('nhtsa_safety_rating_values',))
            self.assertFalse((root / 'final').exists())


if __name__ == "__main__":
    unittest.main()
