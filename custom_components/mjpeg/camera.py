"""Support for IP Cameras."""

import logging
from contextlib import closing

import aiohttp
import requests
# import Exception
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
import voluptuous as vol
from PIL import Image, ImageDraw, ImageFont
import io
import asyncio
from typing import Any, Awaitable, Optional, cast


from aiohttp.hdrs import USER_AGENT, CONTENT_TYPE
from aiohttp import web
from aiohttp.web_exceptions import HTTPGatewayTimeout, HTTPBadGateway
import async_timeout

import  cv2
import numpy as np


from datetime import timedelta
import time
import random

from homeassistant.const import (
    CONF_NAME,
    CONF_USERNAME,
    CONF_PASSWORD,
    CONF_AUTHENTICATION,
    HTTP_BASIC_AUTHENTICATION,
    HTTP_DIGEST_AUTHENTICATION,
    CONF_VERIFY_SSL,
    CONF_TIMEOUT,
)
from homeassistant.core import callback

from homeassistant.components.camera import PLATFORM_SCHEMA, Camera
from . import (
    async_get_clientsession,
    PyDOODS,
)
from homeassistant.helpers import config_validation as cv

_LOGGER = logging.getLogger(__name__)

CONF_MJPEG_URL = "mjpeg_url"
CONF_STILL_IMAGE_URL = "still_image_url"
CONTENT_TYPE_HEADER = "Content-Type"

DEFAULT_NAME = "Mjpeg Camera"
DEFAULT_VERIFY_SSL = True



ATTR_MATCHES = "matches"
ATTR_SUMMARY = "summary"
ATTR_TOTAL_MATCHES = "total_matches"

CONF_URL = "url"
CONF_AUTH_KEY = "auth_key"
CONF_DETECTOR = "detector"
CONF_LABELS = "labels"
CONF_AREA = "area"
CONF_COVERS = "covers"
CONF_TOP = "top"
CONF_BOTTOM = "bottom"
CONF_RIGHT = "right"
CONF_LEFT = "left"
CONF_FILE_OUT = "file_out"

DEFAULT_DETECTOR_URL = "http://192.168.8.104:8088"
DEFAULT_CONF_DETECTOR = "default"

BBOX_UPDATE_INTERVAL = timedelta(seconds=1)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_MJPEG_URL): cv.url,
        vol.Optional(CONF_STILL_IMAGE_URL): cv.url,
        vol.Optional(CONF_AUTHENTICATION, default=HTTP_BASIC_AUTHENTICATION): vol.In(
            [HTTP_BASIC_AUTHENTICATION, HTTP_DIGEST_AUTHENTICATION]
        ),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_PASSWORD): cv.string,
        vol.Optional(CONF_USERNAME): cv.string,
        vol.Optional(CONF_VERIFY_SSL, default=DEFAULT_VERIFY_SSL): cv.boolean,
        vol.Optional(CONF_URL, default=DEFAULT_DETECTOR_URL): cv.string,
        vol.Optional(CONF_DETECTOR,default=DEFAULT_CONF_DETECTOR): cv.string,
        vol.Required(CONF_TIMEOUT, default=90): cv.positive_int,
        vol.Optional(CONF_AUTH_KEY, default=""): cv.string,
    }
)

def get_detectors(config):
    url = config[CONF_URL]
    auth_key = config[CONF_AUTH_KEY]
    detector_name = config[CONF_DETECTOR]
    timeout = config[CONF_TIMEOUT]
    doods = None
    try:
        doods = PyDOODS(url, auth_key, timeout)
        response = doods.get_detectors()
        if not isinstance(response, dict):
            _LOGGER.warning("Could not connect to doods server: %s", url)
        detector = {}
        for server_detector in response["detectors"]:
            if server_detector["name"] == detector_name:
                detector = server_detector
                break
        if not detector:
            _LOGGER.warning(
                "Detector %s is not supported by doods server %s", detector_name, url
            )
            return None
        return doods
    except Exception as e:
        _LOGGER.error(" Connect AI server is failed!! %s",e)
        return None



async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up a MJPEG IP Camera."""
    filter_urllib3_logging()

    doods = get_detectors(config)

    if discovery_info:
        config = PLATFORM_SCHEMA(discovery_info)
    async_add_entities([MjpegCamera(hass, doods ,config)])


def filter_urllib3_logging():
    """Filter header errors from urllib3 due to a urllib3 bug."""
    urllib3_logger = logging.getLogger("urllib3.connectionpool")
    if not any(isinstance(x, NoHeaderErrorFilter) for x in urllib3_logger.filters):
        urllib3_logger.addFilter(NoHeaderErrorFilter())


def extract_image_from_mjpeg(stream):
    """Take in a MJPEG stream object, return the jpg from it."""
    data = b""

    for chunk in stream:
        data += chunk
        jpg_end = data.find(b"\xff\xd9")

        if jpg_end == -1:
            continue

        jpg_start = data.find(b"\xff\xd8")

        if jpg_start == -1:
            continue

        return data[jpg_start : jpg_end + 2]

def make_mjpeg_from_image(image):
    """Take in a jpg , return the MJPEG stream object from it."""
    mjpeg = bytes()
    mjpeg += b"--BoundaryString\r\n"
    mjpeg += b"Content-type: image/jpeg\r\n"
    mjpeg += b"Content-Length:  " + bytes(str(len(image)), encoding="utf8") + b"\r\n\r\n"
    mjpeg += image
    return mjpeg

async def mjpeg_response_write(response, mjpeg):
    size = 4096
    data_list = [mjpeg[i:i + size] for i in range(0, len(mjpeg), size)]
    for data in data_list:
        # _LOGGER.info("new data ,data.len=%s", len(data))
        await response.write(data)


class MjpegCamera(Camera):
    """An implementation of an IP camera that is reachable over a URL."""

    def __init__(self,hass, doods, device_info):
        """Initialize a MJPEG camera."""
        super().__init__()
        self._confidence = 0.3
        self.hass = hass
        self.doods = doods
        self._device_info = device_info
        self._url = device_info[CONF_URL]
        self._auth_key = device_info[CONF_AUTH_KEY]
        self._detector_name = device_info[CONF_DETECTOR]
        self._timeout = device_info[CONF_TIMEOUT]

        #
        # _LOGGER.info("---------------------url=%s,_auth_key=%s,detector_name=%s,_timeout=%s",self._url,self._auth_key,self._detector_name,self._timeout)

        self._name = device_info.get(CONF_NAME)
        self._authentication = device_info.get(CONF_AUTHENTICATION)
        self._username = device_info.get(CONF_USERNAME)
        self._password = device_info.get(CONF_PASSWORD)
        self._mjpeg_url = device_info[CONF_MJPEG_URL]
        self._still_image_url = device_info.get(CONF_STILL_IMAGE_URL)
        self._detectors_response = {}

        self._auth = None
        if self._username and self._password:
            if self._authentication == HTTP_BASIC_AUTHENTICATION:
                self._auth = aiohttp.BasicAuth(self._username, password=self._password)
        self._verify_ssl = device_info.get(CONF_VERIFY_SSL)
        dconfig={'pig':0.2,'person':0.2}

        self._dconfig = dconfig


        self.hass.helpers.event.async_track_time_interval(self.async_update_bbox,BBOX_UPDATE_INTERVAL)
        # self.hass.helpers.event.async_track_time_interval(update_tokens, TOKEN_CHANGE_INTERVAL)

    @callback
    def async_update_bbox(self,time):

        # image = await self.async_camera_image()

        self.hass.async_add_job(self.detector_task)
        # _LOGGER.error("image.len %s",type(image))

    def response_time_delta(self):
        if 'ctdet' in self._detectors_response and  'updatetime' in self._detectors_response['ctdet']:
            now = time.time()
            uptime = self._detectors_response['ctdet']['updatetime']
            delta_time = int(now - uptime)
            # _LOGGER.info("now=%s,uptime=%s------------------------delta_time =  %s",now, uptime ,delta_time)
            return delta_time
        else:
            return int(1000)

    def draw_bbox(self,image):

        if 'ctdet' in self._detectors_response and self.response_time_delta() < 3:

            # _LOGGER.info("response_time_delta =  %s",self.response_time_delta())

            results = self._detectors_response['ctdet']['detections']
            for bbox in results:
                # _LOGGER.info("bbox=%s",bbox)
                if bbox['confidence'] > self._confidence:
                    img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    font = ImageFont.truetype('NotoSansCJK-Black.ttc',18)
                    fillColor = (255,255,0)
                    position = (bbox['top']+6, bbox['left'] + 2)
                    str = bbox['label']+":{}".format(round(bbox['confidence']*100,1))+"%"

                    draw = ImageDraw.Draw(img_PIL)
                    draw.text(position, str, font=font, fill=fillColor)
                    image = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

                    cv2.rectangle(image,
                        (bbox['top'], bbox['left']),
                        (bbox['bottom'], bbox['right']), (255,0,0), 2)
                    # cv2.putText(image, bbox['label'], (bbox['top'], bbox['left'] - 2),
                    #     3, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        else:
            _LOGGER.warn("object detect sever is not ready! ")

        return  image

    async def detector_task(self):

        start = time.time()
        await asyncio.sleep(random.randint(2,5))
        _LOGGER.info("sleep: %s",time.time()-start)
        # _LOGGER.warn("detector_task*******************")

        image = await self.async_camera_image()
        # _LOGGER.debug("image: %s",type(image))
        if self.doods is None:
            _LOGGER.warn("AI is not in server,will try reconnect" )
            self.doods = get_detectors(self._device_info)
        else:

            try:
                # _LOGGER.info("will detect the image!!!,image.len=%s",len(image))
                response = self.doods.detect(image, detector_name=self._detector_name)
                self._detectors_response[response['model']]= response
                _LOGGER.info("self._detectors_response=%s", self._detectors_response)

            except Exception as e:
                _LOGGER.error(" detector_task is failed !! %s", e)

            # await asyncio.sleep(5)
            # response = self.doods.detect(image, detector_name='multi_pose')
            # self._detectors_response[response['model']]= response
            # # _LOGGER.info("self._detectors_response=%s",self._detectors_response)
            #
            # nparry = np.fromstring(image,np.uint8)
            # img = cv2.imdecode(nparry, 3)
            #
            #
            # # _LOGGER.info("self._detectors_response=%s",self._detectors_response)
            #
            # results = self._detectors_response['ctdet']['detections']
            # for bbox in results:
            #     _LOGGER.info("bbox=%s",bbox)
            #     if bbox['confidence'] > self._confidence:
            #         cv2.rectangle(img,
            #               (bbox['top'], bbox['left']),
            #               (bbox['bottom'], bbox['right']), (255,0,0), 2)
            #         cv2.putText(img, bbox['label'], (bbox['top'], bbox['left'] - 2),
            #             3, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                # if bbox[4] > self.opt.vis_thresh:

                    # if bbox[4] > self.opt.vis_thresh:

            # cv2.rectangle(img,
            #               (self._detectors_response[response['model']]['detections'][], bbox[1] - cat_size[1] - 2),
            #               (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            # cv2.putText(img, txt, (bbox[0], bbox[1] - 2),
            #     font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            # cv2.imwrite('image_{}.jpg'.format(random.randint(1,6)),img)


    async def async_camera_image(self):
        """Return a still image response from the camera."""
        # DigestAuth is not supported
        if (
            self._authentication == HTTP_DIGEST_AUTHENTICATION
            or self._still_image_url is None
        ):
            image = await self.hass.async_add_job(self.camera_image)

            return image

        websession = async_get_clientsession(self.hass, verify_ssl=self._verify_ssl)
        try:
            with async_timeout.timeout(10):
                response = await websession.get(self._still_image_url, auth=self._auth)

                image = await response.read()

                return image

        except asyncio.TimeoutError:
            _LOGGER.error("Timeout getting camera image")

        except aiohttp.ClientError as err:
            _LOGGER.error("Error getting new camera image: %s", err)

    def camera_image(self):
        """Return a still image response from the camera."""
        if self._username and self._password:
            if self._authentication == HTTP_DIGEST_AUTHENTICATION:
                auth = HTTPDigestAuth(self._username, self._password)
            else:
                auth = HTTPBasicAuth(self._username, self._password)
            req = requests.get(
                self._mjpeg_url,
                auth=auth,
                stream=True,
                timeout=10,
                verify=self._verify_ssl,
            )
        else:
            req = requests.get(self._mjpeg_url, stream=True, timeout=10)
        # _LOGGER.warn("camera_image*******************")
        # https://github.com/PyCQA/pylint/issues/1437
        # pylint: disable=no-member
        with closing(req) as response:
            return extract_image_from_mjpeg(response.iter_content(102400))



    async def handle_async_mjpeg_stream(self, request):
        """Generate an HTTP MJPEG stream from the camera."""
        # aiohttp don't support DigestAuth -> Fallback
        if self._authentication == HTTP_DIGEST_AUTHENTICATION:
            return await super().handle_async_mjpeg_stream(request)
        _LOGGER.debug("handle_async_mjpeg_stream*******************")
        # connect to stream
        websession = async_get_clientsession(self.hass, verify_ssl=self._verify_ssl)
        stream_coro = websession.get(self._mjpeg_url, auth=self._auth)


        # return await async_aiohttp_proxy_web(self.hass, request, stream_coro)
        return await self.async_aiohttp_proxy_web(request, stream_coro)

    @property
    def name(self):
        """Return the name of this camera."""
        return self._name


    async def async_aiohttp_proxy_stream(
        self,
        request: web.BaseRequest,
        stream: aiohttp.StreamReader,
        content_type: str,
        buffer_size: int = 102400,
        timeout: int = 10,
    ) -> web.StreamResponse:
        """Stream a stream to aiohttp web response."""
        response = web.StreamResponse()
        response.content_type = content_type
        await response.prepare(request)
        buffer = bytes()

        try:

            while True:
                with async_timeout.timeout(timeout):
                    data = await stream.read(buffer_size)
                if not data:
                        break

                buffer += data
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9")

                if start != -1 and end != -1:
                    image = buffer[start:end+2]
                    buffer = buffer[end + 2:]
                    pil_img =  Image.open(io.BytesIO(bytearray(image)))
                    img_width, img_height = pil_img.size


                    _LOGGER.info("img_width=%s, img_height=%s",img_width, img_height)
                    nparry = np.fromstring(image, np.uint8)
                    if len(nparry) > 0:
                        opencv_image = cv2.imdecode(nparry, 3)
                        opencv_image = self.draw_bbox(opencv_image)
                        image_file = 'stream_{}.jpg'.format(random.randint(1, 10))
                        # _LOGGER.info("write file =%s",image_file)
                        # cv2.imwrite(image_file, opencv_image)

                        img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                        _, opencv_image = cv2.imencode('.jpg', opencv_image, img_param)

                        image = opencv_image.tobytes()

                        mjpeg = make_mjpeg_from_image(image)

                        await mjpeg_response_write(response,mjpeg)


                    # if len(nparry) > 0:
                    #     img = cv2.imdecode(nparry, 3)
                    #     cv2.imwrite('stream_{}.jpg'.format(random.randint(1,10)), img)
                    #     img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    #     _, mjpeg = cv2.imencode('.jpg', img, img_param)
                    #
                    #     mjpeg_bytes= mjpeg.tobytes()
                    #     start = mjpeg_bytes.find(b"\xff\xd8")
                    #     end = mjpeg_bytes.find(b"\xff\xd9")
                        # _LOGGER.info("new jpg ,mjpeg.len=%s, mjpeg.len=%s,mjpeg=【start】=%s,mjpeg=【end】=%s", len(mjpeg),len(mjpeg),start, end)





                # await response.write(data)

                    # _LOGGER.info("new data ,data.len=%s", len(data))
                    # buffer += data
                    # start = buffer.find(b"\xff\xd8")
                    # end = buffer.find(b"\xff\xd9")
                    #
                    # if start != -1 and end != -1:
                    #     jpg = buffer[start:end+2]
                    #     _LOGGER.info("new jpg ,jpg.len=%s", len(jpg))
                    #     _LOGGER.info("buffer[0]=%X,buffer[-1]=%X,buffer[-2]=%X,buffer[-3]=%X",buffer[start],buffer[end+2-1],buffer[end+2-2],buffer[end+2-3])
                    #     if len(jpg) > 0:
                    #         _LOGGER.info("jpg[0]=%X,jpg[-1]=%X,jpg[-2]=%X,jpg[-3]=%X", jpg[0], jpg[-1],jpg[-2],jpg[-3])
                    #     # img = Image.open(io.BytesIO(bytearray(jpg)))
                    #     # io.BytesIO(bytearray(jpg))
                    #     # nparry = np.fromstring(jpg,np.uint8)
                    #     # if len(nparry) > 0 and flag%100 ==0:
                    #     #     img = cv2.imdecode(nparry, 3)
                    #     #     cv2.imwrite('stream_{}.jpg'.format(flag),img)
                    #     #     flag += 1
                    #     #
                    #     buffer = buffer[end+2:]
                    #     # _LOGGER.error("buffer.len=%s",len(buffer))
                    #     _LOGGER.error("new frame")

                        # await response.write(jpg)

            # _LOGGER.error("response headers %s",response.headers)
        except (asyncio.TimeoutError, aiohttp.ClientError):
            # Something went wrong fetching data, closed connection
            pass

        return response


    async def async_aiohttp_proxy_web(
            self,
            request: web.BaseRequest,
            web_coro: Awaitable[aiohttp.ClientResponse],
            buffer_size: int = 102400,
            timeout: int = 10,
    ) -> Optional[web.StreamResponse]:
        """Stream websession request to aiohttp web response."""
        _LOGGER.debug("async_aiohttp_proxy_web")
        try:
            with async_timeout.timeout(timeout):
                req = await web_coro

        except asyncio.CancelledError:
            # The user cancelled the request
            return None

        except asyncio.TimeoutError as err:
        # Timeout trying to start the web request
            raise HTTPGatewayTimeout() from err

        except aiohttp.ClientError as err:
        # Something went wrong with the connection
            raise HTTPBadGateway() from err

        try:
            return await self.async_aiohttp_proxy_stream(
                request, req.content, req.headers.get(CONTENT_TYPE)
            )
        finally:
            req.close()



class NoHeaderErrorFilter(logging.Filter):
    """Filter out urllib3 Header Parsing Errors due to a urllib3 bug."""

    def filter(self, record):
        """Filter out Header Parsing Errors."""
        return "Failed to parse headers" not in record.getMessage()
