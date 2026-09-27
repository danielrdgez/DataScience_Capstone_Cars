"""Safety collection regression tests; all HTTP responses are mocked."""

import json
import sqlite3
import unittest
from unittest.mock import patch

import build_car_data_db as importer


def detail(vehicle_id="7520", rating="5"):
    return {"Results": [{"VehicleId": int(vehicle_id), "VehicleDescription": "detail description",
                         "OverallRating": rating, "RolloverRating": "Not Rated",
                         "RolloverPossibility": 0.0, "dynamicTipResult": " ",
                         "FrontCrashDriversideRating": None, "VehiclePicture": ""}]}


class SafetyRatingsTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        importer.create_schema(self.conn)
        self.http = patch.object(importer, "request_json").start()
        self.sleep = patch.object(importer.time, "sleep").start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self.conn.close)

    def seed(self, model="RDX", vehicle_id="7520", status="success"):
        qid = self.conn.execute("""INSERT INTO nhtsa_queries
            (query_type,model_year,make,model,status,result_count)
            VALUES ('safety_ratings',2013,'Acura',?,?,1)""", (model, status)).lastrowid
        for k, v in [("VehicleId", vehicle_id), ("VehicleDescription", "original description")]:
            self.conn.execute('INSERT INTO nhtsa_safety_rating_values VALUES (?,?,?,?)', (qid, vehicle_id, k, v))
        self.conn.commit()
        return qid

    def values(self, qid, vehicle_id="7520"):
        return dict(self.conn.execute("""SELECT field_name,field_value
            FROM nhtsa_safety_rating_values WHERE query_id=? AND vehicle_id=?""", (qid, vehicle_id)))

    def run_fetch(self, model="RDX", refresh=False, attempted=None):
        importer.fetch_safety_ratings(self.conn, 2013, "Acura", model, 0.25,
                                     refresh, set() if attempted is None else attempted)

    def test_backfill_keeps_source_representations_and_existing_identity(self):
        qid = self.seed()
        self.http.return_value = detail()
        self.run_fetch()
        self.http.assert_called_once_with(importer.BASE + "/SafetyRatings/VehicleId/7520", {"format": "json"})
        values = self.values(qid)
        self.assertEqual(values["OverallRating"], "5")
        self.assertEqual(values["RolloverRating"], "Not Rated")
        self.assertEqual(values["RolloverPossibility"], "0.0")
        self.assertEqual(values["dynamicTipResult"], " ")
        self.assertIsNone(values["FrontCrashDriversideRating"])
        self.assertEqual(values["VehiclePicture"], "")
        self.assertEqual(values["VehicleDescription"], "original description")
        cached = json.loads(self.conn.execute('SELECT results_json FROM nhtsa_safety_rating_fetches').fetchone()[0])
        self.assertEqual(cached, detail()["Results"])
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM nhtsa_queries').fetchone()[0], 1)

    def test_new_query_discovers_then_fetches_every_variant(self):
        self.http.side_effect = [{"Results": [
            {"VehicleId": 7520, "VehicleDescription": "FWD"},
            {"VehicleId": 7521, "VehicleDescription": "AWD"}]}, detail(), detail("7521", "4")]
        self.run_fetch()
        qid = self.conn.execute('SELECT query_id FROM nhtsa_queries').fetchone()[0]
        self.assertEqual(self.values(qid)["OverallRating"], "5")
        self.assertEqual(self.values(qid, "7521")["OverallRating"], "4")
        self.assertEqual(self.http.call_count, 3)
        self.assertEqual(self.sleep.call_count, 3)

    def test_resume_and_shared_variant_use_completed_cache(self):
        first = self.seed()
        second = self.seed(model="Shared")
        self.http.return_value = detail()
        self.run_fetch()
        self.run_fetch()
        self.run_fetch(model="Shared")
        self.assertEqual(self.http.call_count, 1)
        self.assertEqual(self.values(first)["OverallRating"], self.values(second)["OverallRating"])

    def test_failure_preserves_fields_and_retries_next_run(self):
        qid = self.seed()
        self.conn.execute('INSERT INTO nhtsa_safety_rating_values VALUES (?,?,?,?)', (qid, "7520", "OverallRating", "4"))
        self.conn.commit()
        self.http.side_effect = RuntimeError("HTTP 503")
        self.run_fetch()
        self.assertEqual(self.values(qid)["OverallRating"], "4")
        self.assertEqual(self.conn.execute('SELECT status FROM nhtsa_safety_rating_fetches').fetchone()[0], "error")
        self.http.side_effect = None
        self.http.return_value = detail()
        self.run_fetch()
        self.assertEqual(self.values(qid)["OverallRating"], "5")
        self.assertEqual(self.conn.execute('SELECT status FROM nhtsa_safety_rating_fetches').fetchone()[0], "success")

    def test_malformed_or_mismatched_details_do_not_mark_complete(self):
        qid = self.seed()
        for response in ({}, {"Results": [{"VehicleId": 7520}]}, detail("9999")):
            with self.subTest(response=response):
                self.http.return_value = response
                self.run_fetch()
                self.assertNotIn("OverallRating", self.values(qid))
                self.assertEqual(self.conn.execute('SELECT status FROM nhtsa_safety_rating_fetches').fetchone()[0], "error")

    def test_zero_results_are_distinct_from_failure_and_cached(self):
        qid = self.seed()
        self.http.return_value = {"Results": []}
        self.run_fetch()
        self.run_fetch()
        self.assertEqual(self.http.call_count, 1)
        self.assertEqual(self.conn.execute('SELECT status FROM nhtsa_safety_rating_fetches').fetchone()[0], "no_results")
        self.assertEqual(len(self.values(qid)), 2)

    def test_one_failed_variant_does_not_block_other_variants(self):
        qid = self.seed()
        for key, value in (("VehicleId", "7521"), ("VehicleDescription", "AWD")):
            self.conn.execute('INSERT INTO nhtsa_safety_rating_values VALUES (?,?,?,?)', (qid, "7521", key, value))
        self.conn.commit()
        self.http.side_effect = [RuntimeError("HTTP 503"), detail("7521", "4")]
        self.run_fetch()
        self.assertNotIn("OverallRating", self.values(qid))
        self.assertEqual(self.values(qid, "7521")["OverallRating"], "4")

    def test_failed_discovery_refresh_keeps_existing_fields(self):
        qid = self.seed()
        before = self.values(qid)
        self.http.side_effect = RuntimeError("HTTP 503")
        self.run_fetch(refresh=True)
        self.assertEqual(self.values(qid), before)
        self.assertEqual(self.conn.execute('SELECT status FROM nhtsa_queries WHERE query_id=?', (qid,)).fetchone()[0], "error")
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM nhtsa_safety_rating_fetches').fetchone()[0], 0)

    def test_refresh_updates_ratings_and_fetches_shared_id_once(self):
        first = self.seed()
        self.seed(model="Shared")
        self.http.return_value = detail(rating="4")
        self.run_fetch()
        self.http.reset_mock()
        self.http.side_effect = [
            {"Results": [{"VehicleId": 7520, "VehicleDescription": "updated"}]}, detail(rating="5"),
            {"Results": [{"VehicleId": 7520, "VehicleDescription": "updated"}]}]
        attempted = set()
        self.run_fetch(refresh=True, attempted=attempted)
        self.run_fetch(model="Shared", refresh=True, attempted=attempted)
        self.assertEqual(self.http.call_count, 3)
        self.assertEqual(self.values(first)["OverallRating"], "5")
        self.assertEqual(self.values(first)["VehicleDescription"], "original description")

    def test_safety_only_dispatch_does_not_request_other_sources(self):
        self.conn.execute('CREATE TABLE used_cars (year TEXT,make_name TEXT,model_name TEXT)')
        self.conn.execute("INSERT INTO used_cars VALUES ('2013','Acura','RDX')")
        self.conn.commit()
        self.http.side_effect = [{"Results": [{"VehicleId": 7520, "VehicleDescription": "FWD"}]}, detail()]
        importer.fetch_vehicle_tables(self.conn, 0.25, kinds=("safety",))
        self.assertTrue(all('/SafetyRatings/' in call.args[0] for call in self.http.call_args_list))
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM nhtsa_recalls').fetchone()[0], 0)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM nhtsa_complaints').fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
