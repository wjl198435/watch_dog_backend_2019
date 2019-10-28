"""The mjpeg component."""
"""Helper for aiohttp webclient stuff."""

import sys
from ssl import SSLContext
from typing import Any, Awaitable, Optional, cast
from typing import Union

import aiohttp
import requests
from aiohttp.hdrs import USER_AGENT, CONTENT_TYPE


from homeassistant.core import callback, Event
from homeassistant.const import EVENT_HOMEASSISTANT_CLOSE, __version__
from homeassistant.helpers.typing import HomeAssistantType
from homeassistant.loader import bind_hass
from homeassistant.util import ssl as ssl_util
import base64
import logging

DATA_CONNECTOR = "aiohttp_connector"
DATA_CONNECTOR_NOTVERIFY = "aiohttp_connector_notverify"
DATA_CLIENTSESSION = "aiohttp_clientsession"
DATA_CLIENTSESSION_NOTVERIFY = "aiohttp_clientsession_notverify"
SERVER_SOFTWARE = "HomeAssistant/{0} aiohttp/{1} Python/{2[0]}.{2[1]}".format(
    __version__, aiohttp.__version__, sys.version_info
)


_LOGGER = logging.getLogger(__name__)

@callback
@bind_hass
def async_get_clientsession(
        hass: HomeAssistantType, verify_ssl: bool = True
) -> aiohttp.ClientSession:
    """Return default aiohttp ClientSession.

    This method must be run in the event loop.
    """
    if verify_ssl:
        key = DATA_CLIENTSESSION
    else:
        key = DATA_CLIENTSESSION_NOTVERIFY

    if key not in hass.data:
        hass.data[key] = async_create_clientsession(hass, verify_ssl)

    return cast(aiohttp.ClientSession, hass.data[key])


@callback
@bind_hass
def async_create_clientsession(
        hass: HomeAssistantType,
        verify_ssl: bool = True,
        auto_cleanup: bool = True,
        **kwargs: Any,
) -> aiohttp.ClientSession:
    """Create a new ClientSession with kwargs, i.e. for cookies.

    If auto_cleanup is False, you need to call detach() after the session
    returned is no longer used. Default is True, the session will be
    automatically detached on homeassistant_stop.

    This method must be run in the event loop.
    """
    connector = _async_get_connector(hass, verify_ssl)

    clientsession = aiohttp.ClientSession(
        loop=hass.loop,
        connector=connector,
        headers={USER_AGENT: SERVER_SOFTWARE},
        **kwargs,
    )

    if auto_cleanup:
        _async_register_clientsession_shutdown(hass, clientsession)

    return clientsession



@callback
def _async_register_clientsession_shutdown(
        hass: HomeAssistantType, clientsession: aiohttp.ClientSession
) -> None:
    """Register ClientSession close on Home Assistant shutdown.

    This method must be run in the event loop.
    """

    @callback
    def _async_close_websession(event: Event) -> None:
        """Close websession."""
        clientsession.detach()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_close_websession)


@callback
def _async_get_connector(
        hass: HomeAssistantType, verify_ssl: bool = True
) -> aiohttp.BaseConnector:
    """Return the connector pool for aiohttp.

    This method must be run in the event loop.
    """
    key = DATA_CONNECTOR if verify_ssl else DATA_CONNECTOR_NOTVERIFY

    if key in hass.data:
        return cast(aiohttp.BaseConnector, hass.data[key])

    if verify_ssl:
        ssl_context: Union[bool, SSLContext] = ssl_util.client_context()
    else:
        ssl_context = False

    connector = aiohttp.TCPConnector(
        loop=hass.loop, enable_cleanup_closed=True, ssl=ssl_context
    )
    hass.data[key] = connector

    async def _async_close_connector(event: Event) -> None:
        """Close connector pool."""
        await connector.close()

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_CLOSE, _async_close_connector)

    return connector



class PyDOODS:
    """A wrapper for the DOODS service"""

    def __init__(self, url, auth_key=None, timeout=90):
        self.url = url
        self.auth_key = auth_key
        self.timeout = timeout
        # Check the connection to DOODS
        self.get_detectors()

    def get_detectors(self):
        """Check the health and return the configured detectors."""
        kwargs = {"timeout": self.timeout}
        if self.auth_key:
            kwargs['headers'] = {'doods-auth-key': self.auth_key}
        response = requests.get(self.url + "/detectors", **kwargs)
        response.raise_for_status()
        return response.json()

    def detect(self, image, dconfig={}, detector_name="default", id=""):
        """Perform a detection and return the results."""
        print("detect begin **************")
        kwargs = {"timeout": self.timeout}
        if self.auth_key:
            kwargs['headers'] = {'doods-auth-key': self.auth_key}
        response = requests.post(
            self.url + "/detect", json={"data": PyDOODS.encode_image(image), "detector_name": detector_name, "detect": dconfig, "id": id}, **kwargs)
        print(response)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def encode_image(image):
        """base64 encode an image stream."""
        return base64.b64encode(image).decode('ascii')
