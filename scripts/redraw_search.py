"""Historical entry point for the GRACE paper renderer."""
if __package__:
    from .render_figures import main
else:
    from render_figures import main

if __name__ == "__main__":
    main()
