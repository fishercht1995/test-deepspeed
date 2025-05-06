# test-deepspeed

vim /home/ubuntu/miniconda3/envs/deepspeed-inference/lib/python3.10/site-packages/torch/utils/cpp_extension.py

def _is_ninja_available():
    try:
        import os
        os.environ['PATH'] += ':/home/ubuntu/miniconda3/envs/deepspeed-inference/bin'
        subprocess.check_output(['ninja', '--version']).decode()
        return True
    except Exception:
        return False