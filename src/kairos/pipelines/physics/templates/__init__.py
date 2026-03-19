"""Template package for fixed simulation scripts.

Each template is a complete, self-contained Python script that runs inside
the Docker sandbox.  The simulation agent loads the appropriate template,
injects the LLM-generated JSON config, and sends the combined script to
the sandbox for execution.
"""
