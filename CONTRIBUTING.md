# How to contribute to pentapy

We are happy about all contributions! :thumbsup:


## Did you find a bug?

- Ensure that the bug was not already reported under
[GitHub issues](https://github.com/GeoStat-Framework/pentapy/issues)
- If the bug wasn't already reported, open a
[new issue](https://github.com/GeoStat-Framework/pentapy/issues) with a clear
description of the problem and if possible with a
[minimal working example](https://en.wikipedia.org/wiki/Minimal_working_example).
- please add the version number to the issue:

```python
import pentapy
print(pentapy.__version__)
```


## Do you have suggestions for new features?

Open a [new issue](https://github.com/GeoStat-Framework/pentapy/issues)
with your idea or suggestion and we'd love to discuss about it.


## Do you want to enhance pentapy or fix something?

- Fork the repo on [GitHub](https://github.com/GeoStat-Framework/pentapy).
- Commit your stuff
- Add yourself to AUTHORS.md (if you want to).
- We use the black code format, please use the script `black --line-length 79 pentapy/` after you have written your code.
- Add some tests of your extension to ``tests/test_pentapy.py``
  - we use [unittest](https://docs.python.org/3/library/unittest.html)
    for our test suite, have a look at their documentation
  - to run the tests you can use [pytest](https://github.com/pytest-dev/pytest)
    with running ``pytest -v tests/`` from the source directory
- Push to your fork and submit a pull request.
