"""Support for IP Cameras."""

import logging
from contextlib import closing

import traceback

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

CONF_DETECTOR_URL = "detector_url"
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
        vol.Optional(CONF_DETECTOR_URL, default=DEFAULT_DETECTOR_URL): cv.string,
        vol.Optional(CONF_DETECTOR,default=DEFAULT_CONF_DETECTOR): cv.string,
        vol.Required(CONF_TIMEOUT, default=90): cv.positive_int,
        vol.Optional(CONF_AUTH_KEY, default=""): cv.string,
    }
)

def get_detectors(config):
    url = config[CONF_DETECTOR_URL]
    auth_key = config[CONF_AUTH_KEY]
    detector_name = config[CONF_DETECTOR]
    timeout = config[CONF_TIMEOUT]
    _LOGGER.info("get_detectors->url=%s detector_name=%s",url,detector_name)
    doods = None
    try:
        doods = PyDOODS(url, auth_key, timeout)
        response = doods.get_detectors()
        _LOGGER.info("get_detectors->response=%s",response)
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
        self._confidence = 0.32
        self._font_hight=20
        self.hass = hass
        self.doods = doods
        self._device_info = device_info
        self._detect_url = device_info[CONF_DETECTOR_URL]
        self._auth_key = device_info[CONF_AUTH_KEY]
        self._detector_name = device_info[CONF_DETECTOR]
        self._timeout = device_info[CONF_TIMEOUT]

        #
        _LOGGER.info("---------------------url=%s,detector_name=%s,_timeout=%s",self._detect_url,self._detector_name,self._timeout)

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
        self._matches = {}
        self._summary = {}
        self._total_matches = 0

        self.hass.helpers.event.async_track_time_interval(self.async_update_bbox,BBOX_UPDATE_INTERVAL)
        # self.hass.helpers.event.async_track_time_interval(update_tokens, TOKEN_CHANGE_INTERVAL)


        # multi_pose conf
        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [3, 5], [4, 6], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [5, 11], [6, 12], [11, 12],
                      [11, 13], [13, 15], [12, 14], [14, 16]]

        self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                   (255, 0, 0), (0, 0, 255), (255, 0, 255),
                   (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                   (255, 0, 0), (0, 0, 255), (255, 0, 255),
                   (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]

        self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255)]
        self.num_joints = 17

    @callback
    def async_update_bbox(self,time):

        # image = await self.async_camera_image()

        self.hass.async_add_job(self.detector_task)
        # _LOGGER.error("image.len %s",type(image))

    def response_time_delta(self,detector_name='ctdet'):
        if detector_name in self._detectors_response and  'updatetime' in self._detectors_response[detector_name]:
            now = time.time()
            uptime = self._detectors_response[detector_name]['updatetime']
            delta_time = int(now - uptime)
            # _LOGGER.info("now=%s,uptime=%s------------------------delta_time =  %s",now, uptime ,delta_time)
            return delta_time
        else:
            return int(1000)

    def get_ctdet_attributes(self):
        matches = {}
        total_matches = 0

        if not self._detectors_response or self._detectors_response['ctdet']['status'] != 0:
            self._matches = matches
            self._total_matches = total_matches

            return

        if 'ctdet' in self._detectors_response and self.response_time_delta() < 3:

            results = self._detectors_response['ctdet']['detections']

            for bbox in results:
                _LOGGER.info("bbox=%s",bbox)
                if bbox['confidence'] > self._confidence:
                    label = bbox["friendly_label"]
                    if label not in matches:
                        matches[label] = []
                    matches[label].append({"score":round(bbox['confidence']*100,1)})
                    total_matches+=1

        self._matches = matches
        self._total_matches = total_matches

    def add_coco_bbox(self, image ,bbox,color=(0,255,0)):
        cv2.rectangle(image,
                      (bbox['top'], bbox['left']),
                      (bbox['bottom'], bbox['right']), color, 2)
        return image

    def add_coco_hp(self, image ,points, img_id='default'):
        points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
        for j in range(self.num_joints):
            cv2.circle(image,(points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
        for j, e in enumerate(self.edges):
            if points[e].min() > 0:
                cv2.line(image, (points[e[0], 0], points[e[0], 1]),
                     (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
                     lineType=cv2.LINE_AA)
        return image

    def draw_bbox(self,image):

        if 'ctdet' in self._detectors_response and self.response_time_delta() < 3:
            results = self._detectors_response['ctdet']['detections']
            img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            font = ImageFont.truetype('NotoSansCJK-Black.ttc',16)
            fillColor = (255,0,0)
            for bbox in results:
                # _LOGGER.info("bbox=%s",bbox)
                if bbox['confidence'] > self._confidence:

                    position = (bbox['top']+6, bbox['left'] + 2)
                    str = bbox['friendly_label']+":{}".format(round(bbox['confidence']*100,1))+"%"
                    draw = ImageDraw.Draw(img_PIL)
                    draw.text(position, str, font=font, fill=fillColor)

            font = ImageFont.truetype('NotoSansCJK-Black.ttc',18)
            hight=48
            widht=12
            position = (widht, hight)
            draw = ImageDraw.Draw(img_PIL)
            str="总数：{}".format(self._total_matches)
            draw.text(position, str, font=font, fill=(255,0,0))
            hight += self._font_hight

            for label, values in self._matches.items():
                str = "{}:{}".format(label,len(values))
                position = (widht, hight)
                draw.text(position, str, font=font, fill=(255,0,0))
                hight += self._font_hight

            image = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
            for bbox in results:
                if bbox['confidence'] > self._confidence:
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
        await asyncio.sleep(random.randint(1,2))
        # _LOGGER.info("sleep: %s",time.time()-start)
        # _LOGGER.warn("detector_task*******************")

        image = await self.async_camera_image()
        # _LOGGER.debug("image: %s",type(image))
        if self.doods is None:
            _LOGGER.warn("AI is not in server,will try reconnect" )
            self.doods = get_detectors(self._device_info)
        else:

            try:
                _LOGGER.info("will detect the image!!!,detect.url=%s,self._detector_name=%s",self._detect_url,self._detector_name)

                response = self.doods.detect(image, detector_name=self._detector_name)
                self._detectors_response[response['model']]= response
                _LOGGER.info("self._detectors_response=%s", self._detectors_response)

                #获取ctdet model 检测结果
                if response['model'] =='ctdet':
                    _LOGGER.info("response['model'] =='ctdet'")
                    self.get_ctdet_attributes()
                elif response['model'] =='multi_pose':

                    _LOGGER.info("response['model'] =='multi_pose'")


            except Exception as e:
                # _LOGGER.error(" detector_task is failed !! %s", str(e))
                _LOGGER.error(" detector_task is failed !! %s", traceback.print_exc())


    @property
    def device_state_attributes(self):
        # _LOGGER.info("device_state_attributes----->>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return {
            ATTR_MATCHES: self._matches,
            ATTR_SUMMARY :{
                label: len(values) for label, values in self._matches.items()
            },
            ATTR_TOTAL_MATCHES : self._total_matches,
        }

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

                    # _LOGGER.info("img_width=%s, img_height=%s",img_width, img_height)
                    nparry = np.fromstring(image, np.uint8)
                    if len(nparry) > 0:
                        opencv_image = cv2.imdecode(nparry, 3)
                        _LOGGER.info("%s",self._detector_name)
                        if self._detector_name == 'ctdet':
                            _LOGGER.info("async_aiohttp_proxy_stream--->self._detector_name == 'ctdet'")
                            opencv_image = self.draw_bbox(opencv_image)
                        elif self._detector_name == 'multi_pose':
                            _LOGGER.info("self._detector_name == 'multi_pose' ")
                            if 'multi_pose' in self._detectors_response and self.response_time_delta(detector_name=self._detector_name) < 3:

                                detections = self._detectors_response['multi_pose']['detections']
                                # _LOGGER.info("self._detector_name == 'multi_pose' %s ",detections)
                                for detection in detections:
                                    _LOGGER.info("async_aiohttp_proxy_stream--->detection = %s",detection)
                                    bbox = detection['bbox']
                                    keypoints = detection['keypoints']

                                    if bbox['confidence'] > self._confidence:
                                        opencv_image = self.add_coco_bbox(opencv_image,bbox)
                                        opencv_image = self.add_coco_hp(opencv_image,keypoints)



                        # image_file = 'stream_{}.jpg'.format(random.randint(1, 10))
                        # _LOGGER.info("write file =%s",image_file)
                        # cv2.imwrite(image_file, opencv_image)
                        img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        _, opencv_image = cv2.imencode('.jpg', opencv_image, img_param)

                        image = opencv_image.tobytes()
                        mjpeg = make_mjpeg_from_image(image)
                        await mjpeg_response_write(response,mjpeg)

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
