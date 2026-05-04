import os
import platform
import lit.formats

config.name = 'cmlir'
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.c', '.cpp', '.h', '.mlir']
config.excludes = ['CMakeLists.txt', 'lit.cfg.py', 'lit.site.cfg.py', 'Inputs']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.cmlir_obj_root, 'tests')

path_components = []
if hasattr(config, 'cmlir_tools_dir') and config.cmlir_tools_dir:
    path_components.append(config.cmlir_tools_dir)
    
if hasattr(config, 'mlir_tools_dir') and config.mlir_tools_dir:
    path_components.append(config.mlir_tools_dir)
    
if hasattr(config, 'llvm_tools_dir') and config.llvm_tools_dir:
    path_components.append(config.llvm_tools_dir)

if 'PATH' in os.environ:
    path_components.append(os.environ['PATH'])

config.environment = {
    'PATH': os.pathsep.join(path_components),
}

for var in ['HOME', 'TMPDIR', 'TMP', 'TEMP']:
    if var in os.environ:
        config.environment[var] = os.environ[var]

config.substitutions.append(('%cmlir', 'cmlirc'))
config.substitutions.append(('%cmlir', 'chwc'))

tools = {
    'FileCheck': 'FileCheck',
    'not': 'not',
    'count': 'count',
}

for name, tool in tools.items():
    config.substitutions.append(('%' + name, tool))
