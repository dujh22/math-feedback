import subprocess

def run_python(code, timeout=None):
    result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=timeout)
    stdout = result.stdout
    stderr = result.stderr
    return stdout, stderr

if __name__ == '__main__':
    code = 'print("Hello, World!")'
    stdout, stderr = run_python(code)
    print(stdout)
    print(stderr)