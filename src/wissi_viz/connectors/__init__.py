"""API connectors. These are *stubs*: subclass and fill in ``image_info`` / ``tile`` with calls to
your own authenticated client; wissi-viz handles pyramid mapping, tile overlap/padding, caching,
retries, concurrency limits, async clients, annotation conversion and rendering."""
from ..sources.remote import RemoteError, RemoteImageInfo, RemotePyramid, RemoteSlideSource, RetryAfter, TileNotFound
from .base import AnnotationPayload, RemoteConnection, to_layer
from .halolink import HaloLinkConnection
from .pathcore import PathCoreConnection

__all__ = ['AnnotationPayload', 'HaloLinkConnection', 'PathCoreConnection', 'RemoteConnection', 'RemoteError',
           'RemoteImageInfo', 'RemotePyramid', 'RemoteSlideSource', 'RetryAfter', 'TileNotFound', 'to_layer']
