import collections
import math
import os
import re


def method_Key(method: str) -> int:
    method = method.lower()
    if method == 'get':
        return 0
    elif method == 'post':
        return 1
    elif method == 'put':
        return 2
    elif method == 'delete':
        return 3
    elif method == 'head':
        return 4
    elif method == 'patch':
        return 5
    else:
        return 6


def countSpecialCharacter(text: str) -> int:
    pattern = r'[!@#$%^&*()_+{}\[\]:;<>,.?\/\\\-]'
    special_characters = re.findall(pattern, text)
    return len(special_characters)


class RequestStandarization:
    def __init__(self, request: dict):
        self.malicious_keywords = []
        self.request = request

        self.query = request.get('query', {})
        if isinstance(self.query, str) and self.query.find('=') > 0:
            self.query = dict([i.split('=') for i in self.query.split('&')])
        elif not isinstance(self.query, dict):
            self.query = {}

        self.body = request.get('body', {})
        if isinstance(self.body, str):
            self.body = dict([i.split('=') for i in self.body.split('&')])
        elif not isinstance(self.body, dict):
            self.body = {}

        self.headers = request.get('headers', {})
        if isinstance(self.headers, str):
            self.headers = dict([i.split(': ') for i in self.headers.split('\n')])
        elif not isinstance(self.headers, dict):
            self.headers = {}

        self.path = request.get('path', '')
        self.method = request.get('method', 'GET')
        self.__getKeywordMalicious()

    def __countMaliciousKeywords(self, text: str) -> int:
        length = 0
        for word in self.malicious_keywords:
            # add escape character to special characters
            word = re.sub(r'([!@#$%^&*()_+{}\[\]:;<>,.?/\\\-])', r'\\\1', word)
            pattern = r'{}'.format(word.lower())
            length += len(re.findall(pattern, text.lower()))
        return length

    def __getKeywordMalicious(self):
        file_path = os.path.join(os.path.dirname(__file__), 'malicious_keyword')
        with open(file_path, 'r', encoding='utf-8') as file:
            self.malicious_keywords = file.read().splitlines()

    def __countRequestLength(self) -> int:
        return len(str(self.request))

    def __countQueryParams(self) -> int:
        length = 0
        for key, value in self.query.items():
            length += len(value)
        return length

    def __countAcceptEncoding(self) -> int:
        return len(self.headers.get('‘Accept-Encoding', ''))

    def __countAcceptLanguage(self) -> int:
        return len(self.headers.get('Accept-Language', ''))

    def __contentLength(self) -> int:
        return int(self.headers.get('Content-Length', '0'))

    def __countHost(self) -> int:
        return len(self.headers.get('Host', ''))

    def __countUserAgent(self) -> int:
        return len(self.headers.get('User-Agent', ''))

    def __countNumberOfQueryParams(self) -> int:
        length = 0
        for _ in self.query.items():
            length += 1
        return length

    def __countDigitOfQueryParams(self) -> int:
        length = 0
        for key, value in self.query.items():
            length += sum(c.isdigit() for c in value)
        return length

    def __countSpecialCharsOfQueryParams(self) -> int:
        length = 0
        for key, value in self.query.items():
            length += countSpecialCharacter(value)
        return length

    def __countLettersOfQueryParams(self) -> int:
        length = 0
        for key, value in self.query.items():
            length += sum(c.isalpha() for c in value)
        return length

    def __countNumberOfBodyParams(self) -> int:
        length = 0
        for _ in self.body.items():
            length += 1
        return length

    def __countDigitOfBodyParams(self) -> int:
        length = 0
        for key, value in self.body.items():
            length += sum(c.isdigit() for c in value)
        return length

    def __countSpecialCharsOfBodyParams(self) -> int:
        length = 0
        for key, value in self.body.items():
            length += countSpecialCharacter(value)
        return length

    def __countLettersOfBodyParams(self) -> int:
        length = 0
        for key, value in self.body.items():
            length += sum(c.isalpha() for c in value)
        return length

    def __countPathLength(self) -> int:
        return len(self.path)

    def __countSpecialCharsOfPath(self) -> int:
        return countSpecialCharacter(self.path)

    def __countDigitsOfPath(self) -> int:
        return sum(c.isdigit() for c in self.path)

    def __countLettersOfPath(self) -> int:
        return sum(c.isalpha() for c in self.path)

    def __methodIdentifier(self) -> int:
        return method_Key(self.method)

    def __countKeywordinPath(self) -> int:
        return self.__countMaliciousKeywords(self.path)

    def __countKeywordinQuery(self) -> int:
        length = 0
        for key, value in self.query.items():
            length += self.__countMaliciousKeywords(value)
        return length

    def __countKeywordinBody(self) -> int:
        length = 0
        for key, value in self.body.items():
            length += self.__countMaliciousKeywords(value)
        return length

    def __countLengthofCookie(self) -> int:
        return len(self.headers.get('Cookie', ''))

    def __countAccept(self) -> int:
        return len(self.headers.get('Accept', ''))

    def __countAcceptCharset(self) -> int:
        return len(self.headers.get('Accept-Charset', ''))

    def __countContentType(self) -> int:
        return len(self.headers.get('Content-Type', ''))

    def __countReferer(self) -> int:
        return len(self.headers.get('Referer', ''))

    def __entropyRequest(self) -> float:
        v = str(self.request)
        return (-1) * sum(
            i / len(v) * math.log2(i / len(v))
            for i in collections.Counter(v).values())

    def getFeatures(self) -> list:
        return [
            self.__countRequestLength(),
            self.__countQueryParams(),
            self.__countAcceptEncoding(),
            self.__countAcceptLanguage(),
            self.__contentLength(),
            self.__countUserAgent(),
            self.__countNumberOfQueryParams(),
            self.__countDigitOfQueryParams(),
            self.__countSpecialCharsOfQueryParams(),
            self.__countLettersOfQueryParams(),
            self.__countNumberOfBodyParams(),
            self.__countDigitOfBodyParams(),
            self.__countSpecialCharsOfBodyParams(),
            self.__countLettersOfBodyParams(),
            self.__countPathLength(),
            self.__countSpecialCharsOfPath(),
            self.__countDigitsOfPath(),
            self.__countLettersOfPath(),
            self.__methodIdentifier(),
            self.__countKeywordinPath(),
            self.__countKeywordinQuery(),
            self.__countKeywordinBody(),
            self.__countLengthofCookie(),
            self.__countAccept(),
            self.__countAcceptCharset(),
            self.__countContentType(),
            self.__countReferer(),
            self.__entropyRequest(),
        ]
