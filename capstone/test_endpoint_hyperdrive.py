import urllib.request
import json
import os
import ssl


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(
    True
)  # this line is needed if you use self-signed certificate in your scoring service.


data = {
    "data": [
        [
            11232.723333333333,
            0.46631297568480173,
            77.93616381327311,
            499.4271993307067,
            597.4316278993895,
            1433.1934792655031,
            1812.9871746475746,
            1221.3179555693464,
            15.276443349875432,
            11.160192321851317,
            15.509079543020755,
            19.497259931350946,
            321.4541201749575,
            326.61349442250025,
            12.119224496040358,
            2163.894106372574,
            8574.320839032911,
            3.736963274994671,
            2790.2876388377267,
            342.00365908115236,
            60.94931157595957,
            2355.346713713548,
            222.11019162224832,
            222.11011762483867,
            25.745507644434074,
            15.447304586660438,
            209.7010773417488,
            221.91712952018176,
            16.816921044395563,
            9.828378554568044,
            25.35287563894261,
            41.848948800547895,
        ],
        [
            13408.653333333334,
            0.48565649588902793,
            78.9000242360433,
            493.0656137124189,
            590.5507651892489,
            1418.7055884738722,
            1795.0431685356252,
            1205.1466903260578,
            14.22116686593009,
            10.367926013490974,
            14.437732858812279,
            18.18713046818446,
            300.3370563872515,
            305.1473457352332,
            11.193799048996665,
            2158.749451989125,
            8528.30772719627,
            3.47049089141952,
            2764.853138867338,
            319.5260164766841,
            56.94390526628915,
            2215.04850194227,
            208.7797151215269,
            208.77955258939565,
            24.170834649622364,
            14.50250078977343,
            196.83722326673205,
            208.3200957300851,
            17.001838506636872,
            9.814327984230403,
            25.414812136510168,
            41.597969674987226,
        ],
    ]
}

body = str.encode(json.dumps(data))

url = (
    "http://22404b60-4e70-405d-b5fb-9dd307814897.southcentralus.azurecontainer.io/score"
)
api_key = "1wHnVk4seZABGYV6TOzNY3gThBt4TMes"
headers = {"Content-Type": "application/json", "Authorization": ("Bearer " + api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", "ignore")))
