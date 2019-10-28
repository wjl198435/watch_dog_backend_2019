# """Example Load Platform integration."""
DOMAIN = 'image_processing'
# #
# #
def setup(hass, config):
    """Your controller/hub specific code."""
    # Data that you want to share with your platforms
    # hass.data[DOMAIN] = {
    #     'temperature': 23
    # }

    hass.helpers.discovery.load_platform('ctdet', DOMAIN, {}, config)

    return True
