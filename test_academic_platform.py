import unittest
import json
import app

class TestAcademicAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    def test_routes_status(self):
        routes = ['/', '/login', '/register', '/academic', '/api/academic/overview', '/api/academic/states', '/api/academic/categories', '/api/academic/schema']
        for r in routes:
            res = self.client.get(r)
            self.assertEqual(res.status_code, 200, f"Route {r} failed with {res.status_code}")

    def test_classification_single(self):
        res = self.client.post('/api/academic/run', json={
            'analysis_type': 'classification',
            'model': 'logistic_regression',
            'parameters': {'test_size': 0.2, 'C': 1.0}
        })
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertTrue(d['success'])
        self.assertIn('accuracy', d['metrics'])
        self.assertIn('roc_curve', d)
        self.assertIn('pr_curve', d)
        self.assertIn('feature_importance', d)
        self.assertIn('class_distribution', d)
        self.assertIn('confusion_matrix', d)

    def test_classification_compare_all(self):
        res = self.client.post('/api/academic/run', json={
            'analysis_type': 'classification',
            'model': 'compare_all',
            'parameters': {'test_size': 0.2}
        })
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertTrue(d['success'])
        self.assertTrue(len(d['comparison']) >= 4)

    def test_regression_single(self):
        res = self.client.post('/api/academic/run', json={
            'analysis_type': 'regression',
            'model': 'ridge',
            'parameters': {'test_size': 0.2, 'alpha': 1.0}
        })
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertTrue(d['success'])
        self.assertIn('r2', d['metrics'])
        self.assertIn('actual_vs_predicted', d)
        self.assertIn('residuals', d)
        self.assertIn('coefficients', d)

    def test_regression_compare_all(self):
        res = self.client.post('/api/academic/run', json={
            'analysis_type': 'regression',
            'model': 'compare_all',
            'parameters': {'test_size': 0.2}
        })
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertTrue(d['success'])
        self.assertTrue(len(d['comparison']) >= 5)

    def test_clustering_kmeans(self):
        res = self.client.post('/api/academic/run', json={
            'analysis_type': 'clustering',
            'model': 'kmeans',
            'parameters': {'n_clusters': 4, 'sample_size': 1000}
        })
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertTrue(d['success'])
        self.assertIn('silhouette_score', d['metrics'])
        self.assertIn('elbow', d['extra'])
        self.assertIn('silhouettes', d['extra'])
        self.assertIn('scatter_data', d)
        self.assertIn('cluster_profiles', d)

    def test_clustering_agglomerative(self):
        res = self.client.post('/api/academic/run', json={
            'analysis_type': 'clustering',
            'model': 'agglomerative',
            'parameters': {'n_clusters': 3, 'sample_size': 500}
        })
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertTrue(d['success'])
        self.assertIn('dendrogram', d['extra'])

    def test_association_rules(self):
        res = self.client.post('/api/academic/run', json={
            'analysis_type': 'association_rules',
            'model': 'apriori',
            'parameters': {'min_support': 0.00005, 'min_confidence': 0.01}
        })
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertTrue(d['success'])
        self.assertIn('rules', d)
        self.assertIn('frequent_itemsets', d)

    def test_association_rules_empty(self):
        res = self.client.post('/api/academic/run', json={
            'analysis_type': 'association_rules',
            'model': 'apriori',
            'parameters': {'min_support': 0.9, 'min_confidence': 0.99}
        })
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertTrue(d['success'])
        self.assertTrue(d['no_rules'])

    def test_olap_operations(self):
        for op, params in [
            ('rollup', {'dimension': 'time', 'level': 'year'}),
            ('drilldown', {'year': '2018'}),
            ('slice', {'dimension': 'geography', 'state': 'SP'}),
            ('dice', {'states': ['SP', 'RJ'], 'years': [2018]})
        ]:
            res = self.client.post('/api/academic/run', json={
                'analysis_type': 'olap',
                'model': op,
                'parameters': params
            })
            self.assertEqual(res.status_code, 200)
            d = res.get_json()
            self.assertTrue(d['success'])
            self.assertIn('data', d)
            self.assertIn('query_representation', d)

    def test_data_exploration(self):
        res = self.client.get('/api/academic/exploration?dataset=churn&feature=monetary')
        self.assertEqual(res.status_code, 200)
        d = res.get_json()
        self.assertIn('feature_distribution', d)
        self.assertIn('missing_stats', d)
        self.assertIn('desc_stats', d)

if __name__ == '__main__':
    unittest.main()
