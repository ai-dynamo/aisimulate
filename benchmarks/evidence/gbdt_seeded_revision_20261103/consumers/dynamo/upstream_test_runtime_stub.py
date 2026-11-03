"""Only unavailable Dynamo native-runtime logging is suppressed for upstream tests."""
import sys
import types

runtime_logging = types.ModuleType("dynamo.runtime.logging")
runtime_logging.configure_dynamo_logging = lambda *args, **kwargs: None
sys.modules["dynamo.runtime.logging"] = runtime_logging
