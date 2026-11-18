import unittest

from sglang_simulator.native_replay import simulation_server_args


class NativeReplayArgsTests(unittest.TestCase):
    def test_resolved_template_uses_selected_model_and_token_ids(self):
        template = dict(model_path='/real/model', tokenizer_path='/real/tokenizer',
            language_model_only=False, chunked_prefill_size=2048, obsolete_field=1)
        case = dict(model_path='/local/metadata', chunked_prefill_size=8192)
        fields = {'model_path', 'tokenizer_path', 'language_model_only', 'chunked_prefill_size'}
        kwargs = simulation_server_args(case, template, fields)
        self.assertEqual(kwargs['model_path'], '/local/metadata')
        self.assertEqual(kwargs['tokenizer_path'], '/local/metadata')
        self.assertEqual(kwargs['chunked_prefill_size'], 8192)
        self.assertTrue(kwargs['language_model_only'])
        self.assertTrue(kwargs['skip_tokenizer_init'])
        self.assertNotIn('obsolete_field', kwargs)
        self.assertEqual(template['tokenizer_path'], '/real/tokenizer')

    def test_older_runtime_keeps_supported_fields(self):
        kwargs = simulation_server_args({'model_path': '/local/model'}, {}, {'model_path', 'tokenizer_path'})
        self.assertNotIn('language_model_only', kwargs)
        self.assertEqual(kwargs['tokenizer_path'], '/local/model')
