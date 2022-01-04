from src.data.processor import Preprocessor


@Preprocessor.register('dummy')
def dummy_preproc(sample, test_args=False):
    print(test_args)
    sample['x'] = test_args
    return sample
