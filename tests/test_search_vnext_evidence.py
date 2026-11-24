"""Historical standalone test command."""
if __name__ == "__main__":
    from pathlib import Path
    import runpy
    runpy.run_path(str(Path(__file__).with_name("test_search_evidence.py")), run_name="__main__")
