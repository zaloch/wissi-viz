"""pytest plugin for the Windows crash probe: read cell tables with pyarrow instead of Polars."""

from wissi_viz import objects

objects._engine = lambda engine: 'pyarrow' if engine in ('auto', 'polars') else engine
