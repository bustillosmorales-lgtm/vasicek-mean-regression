"""Downloads the eight bounded-response data sets of the R package betareg
(Cribari-Neto and Zeileis, 2010) from the Rdatasets mirror and checks each file
against the SHA-256 of the copy used for Table 13."""
import hashlib, os, urllib.request

URL = 'https://vincentarelbundock.github.io/Rdatasets/csv/betareg/%s.csv'
SHA = {
    'CarTask': '36570629be3c95e3bdbfd8ab2355506ec6e65d7472e3bf1bca99af896170683a',
    'FoodExpenditure': 'fe02f482d42d75915dadeaff5c60af3bc39e2a0cf0f06452fdb3b2e07e9984f1',
    'GasolineYield': '2bd78e096a64794e26b9f15846410328ad08dd3bc4956fb3f883b628295e14c9',
    'ImpreciseTask': 'a29479f2275499a2cce8b591b11c9acd45d4a7310dea9872d85dc38a756f20ba',
    'MockJurors': 'badf0f30687a119abc9836f55a2a821585c11d64e4955739d02b2381df56804c',
    'ReadingSkills': '1575a7c862c1e4aa48aee3e202912c3cf6bc0d08ebf0f909f0a941ac7935a962',
    'StressAnxiety': '5be004806662d8800ab2e1b7d78f18fbcd7c6285f1be996b60b420b5b8ad2d64',
    'WeatherTask': '3549df262efa658f0d06d9376d4cbdfc682d27344a40165265397e2b7ea05d76',
}

if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(out, exist_ok=True)
    for name, sha in SHA.items():
        raw = urllib.request.urlopen(URL % name).read()
        got = hashlib.sha256(raw).hexdigest()
        if got != sha:
            raise SystemExit('%s: checksum %s differs from %s' % (name, got, sha))
        open(os.path.join(out, name + '.csv'), 'wb').write(raw)
        print('ok', name)
