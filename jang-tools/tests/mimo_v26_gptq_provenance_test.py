"""CPU-only rejection checks for resumable sequential GPTQ artifacts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).parents[1] / 'jang_tools/mimo_v2/v26_gptq_provenance.py'
spec = importlib.util.spec_from_file_location('gptq_provenance', MODULE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class SequentialProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / 'source'; self.src.mkdir()
        self.codes = self.root / 'codes'; self.codes.mkdir()
        self.write(self.src / 'config.json', {'num_hidden_layers':1,'n_routed_experts':2,'moe_layer_freq':[1]})
        self.write(self.src / 'model.safetensors.index.json', {'weight_map':{}})
        stats = self.root / 'stats'; stats.write_bytes(b'fixture statistics')
        self.plan = {'schema':'mimo-v26-jang-plan-v1','stats':str(stats),'expert_default':{'mode':'affine','bits':2,'group_size':128}}
        self.write(self.codes / 'gptq_run.json', {
            'schema':'mimo-v26-gptq-run-v1','recipe_sha256':p.recipe_sha256(self.plan),
            'stats_sha256':p.file_sha256(stats),'tokens_sha256':'a'*64,
            'source_config_sha256':p.file_sha256(self.src / 'config.json'),
            'source_index_sha256':p.file_sha256(self.src / 'model.safetensors.index.json')})
        names=['v26_sequential_gptq.py','v26_gptq.py','v26_quant.py','v26_model.py','v26_source.py','v26_sweep.py']
        self.write(self.codes / 'sequential_run.json', {
            'schema':'mimo-v26-sequential-gptq-v1','max_layers':0,'tokens_sha256':'a'*64,
            'source_files_sha256':{n:p.file_sha256(MODULE.with_name(n)) for n in names},
            'incumbent_manifest_sha256':'b'*64,'grid':'fixed imatrix BF16',
            'propagation':'actual quantized prefix','selection':'minimum of incumbent, RTN and new GPTQ'})
        self.write(self.codes / 'hessian_capture_report.json', {'0':{'completed':True}})
        report={}
        for proj in ['gate_proj','up_proj','down_proj']:
            name='L0.'+proj; path=self.codes / (name+'.safetensors')
            path.write_bytes(b'checksum fixture only; no tensor-layout claim')
            report[name]={'spec':self.plan['expert_default'],'experts':2,'sha256':p.file_sha256(path)}
        self.write(self.codes / 'gptq_report.json', report)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def change(self, name, key, value):
        path=self.codes / name; content=json.loads(path.read_text());content[key]=value;self.write(path,content)

    def test_valid_run_and_method_description(self):
        p.validate_conversion(self.codes,self.plan,self.src)
        metadata=p.describe_run(self.codes)
        self.assertEqual(metadata['propagation'],'actual quantized prefix')
        self.assertIn('incumbent',metadata['guard'])
        self.assertEqual(metadata['report'],'quantization/gptq_report.json')

    def test_corrupted_projection_rejected(self):
        (self.codes/'L0.up_proj.safetensors').write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            p.validate_conversion(self.codes,self.plan,self.src)

    def test_smoke_run_rejected(self):
        self.change('sequential_run.json','max_layers',2)
        with self.assertRaisesRegex(ValueError,'incomplete sequential'):
            p.validate_conversion(self.codes,self.plan,self.src)

    def test_incomplete_capture_rejected(self):
        self.change('hessian_capture_report.json','0',{'completed':False})
        with self.assertRaisesRegex(ValueError,'capture is incomplete'):
            p.validate_conversion(self.codes,self.plan,self.src)

    def test_changed_capture_tokens_rejected(self):
        self.change('sequential_run.json','tokens_sha256','c'*64)
        with self.assertRaisesRegex(ValueError,'token provenance'):
            p.validate_conversion(self.codes,self.plan,self.src)

    def test_changed_implementation_rejected(self):
        path=self.codes/'sequential_run.json';data=json.loads(path.read_text())
        data['source_files_sha256']['v26_quant.py']='c'*64;self.write(path,data)
        with self.assertRaisesRegex(ValueError,'implementation changed'):
            p.validate_conversion(self.codes,self.plan,self.src)

    def test_legacy_run_unchanged(self):
        (self.codes/'sequential_run.json').unlink()
        p.validate_conversion(self.codes,self.plan,self.src)
        metadata=p.describe_run(self.codes)
        self.assertEqual(metadata['propagation'],'source activations between layers')
        self.assertNotIn('incumbent_manifest_sha256',metadata)


if __name__=='__main__':unittest.main()
