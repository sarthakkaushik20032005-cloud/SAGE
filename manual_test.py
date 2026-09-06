import config
from code_executor.sandbox import DockerSandbox

sb = DockerSandbox(config.SANDBOX)

code = '''numbers = [10, 5, 10, 8, 3]
unique = sorted(set(numbers))
print("Second largest:", unique[-2])
'''

res = sb.run(code)

print("STDOUT:")
print(res.stdout)

print("STDERR:")
print(res.stderr)

print("Exit code:", res.exit_code)
print("Succeeded:", res.succeeded)
print("Timed out:", res.timed_out)