"""Template for `python scripts/wissi_probe.py connector_factory_example:make_connection --images ID1,ID2`.

Point the probe at the SAME connector class your app (wissi) uses for HALO Link / PathCore, built with the same
authenticated client — then the probe measures exactly what production will do. Copy this file next to your code,
replace the body, and pass settings with --connector-arg key=value (they arrive as keyword arguments).
"""


def make_connection(**kw):
    # from wissi.integrations.halo import WissiHaloLink, make_client        # ← your app's own modules
    # client = make_client(base_url=kw.get('url') or os.environ['HALO_LINK_URL'], token=os.environ['HALO_LINK_TOKEN'])
    # return WissiHaloLink(client)
    raise SystemExit(
        'connector_factory_example.make_connection: replace the body with your connector (see the comments)'
    )
