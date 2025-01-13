#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# /// script
# requires-python = ">=2.6"
# dependencies = [
#   "requests",
# ]
# ///

"""Automatic assignment handler for Mlucas, GpuOwl/PRPLL, CUDALucas, CUDAPm1, mfaktc, mfakto and cofact.

[*] Python can be downloaded from https://www.python.org/downloads/
    * An .exe version of this script (not requiring Python) can be downloaded from:
        https://download.mersenne.ca/AutoPrimeNet/

[*] Authorship:
     * # EWM: adapted from https://github.com/MarkRose/primetools/blob/master/mfloop.py
            by teknohog and Mark Rose, with help from Gord Palameta.
     * # 2020: revised for CUDALucas by Teal Dulcet and Daniel Connelly
     * # 2020: support for computer registration and assignment-progress via
            direct Primenet-v5-API calls by Loïc Le Loarer <loic@le-loarer.org>
     * # 2024: support for mfaktc added by Tyler Busby and Teal Dulcet

[*] List of supported v5 operations:
    * Update Computer Info (uc, register_instance) (Credit: Loarer & Dulcet)
    * Program Options (po, program_options) (Credit: Connelly & Dulcet)
    * Get Assignment (ga, get_assignment) (Credit: Connelly & Dulcet)
    * Register Assignment (ra, register_assignment) (Credit: Dulcet)
    * Assignment Un-Reserve (au, assignment_unreserve) (Credit: Dulcet)
    * Assignment Progress (ap, send_progress) (Credit: Loarer & Dulcet)
    * Assignment Result (ar, report_result) (Credit: Loarer & Dulcet)
    * Benchmark Data Statistics (bd) N/A
    * Ping Server (ps, ping_server) (Credit: Dulcet)
"""

################################################################################
#                                                                              #
#   (C) 2017-2024 by Daniel Connelly and Teal Dulcet.                          #
#                                                                              #
#  This program is free software; you can redistribute it and/or modify it     #
#  under the terms of the GNU General Public License as published by the       #
#  Free Software Foundation; either version 2 of the License, or (at your      #
#  option) any later version.                                                  #
#                                                                              #
#  This program is distributed in the hope that it will be useful, but WITHOUT #
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       #
#  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for   #
#  more details.                                                               #
#                                                                              #
#  You should have received a copy of the GNU General Public License along     #
#  with this program; see the file GPL.txt.  If not, you may view one at       #
#  http://www.fsf.org/licenses/licenses.html, or obtain one by writing to the  #
#  Free Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA     #
#  02111-1307, USA.                                                            #
#                                                                              #
################################################################################
from __future__ import division, print_function, unicode_literals

import atexit
import ctypes
import errno
import getpass
import glob
import io
import json
import locale
import logging.handlers
import math
import mimetypes
import optparse
import os
import platform
import random
import re
import shutil
import smtplib
import struct
import sys
import tempfile
import textwrap
import threading
import time
import timeit
import uuid
import zipfile
from array import array
from collections import deque
from ctypes.util import find_library
from datetime import datetime, timedelta
from decimal import Decimal
from email import charset, encoders
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, parseaddr
from hashlib import md5
from itertools import chain, count, starmap

try:
	# Python 2
	from future_builtins import map, zip  # ascii, filter, hex, oct
except ImportError:
	pass

try:
	# Python 3+
	from urllib.parse import urlencode
except ImportError:
	from urllib import urlencode

try:
	# Python 3+
	from configparser import ConfigParser
	from configparser import Error as ConfigParserError
except ImportError:
	from ConfigParser import Error as ConfigParserError
	from ConfigParser import SafeConfigParser as ConfigParser

if sys.version_info >= (3, 7):
	# Python 3.7+
	# If is OK to use dict in 3.7+ because insertion order is guaranteed to be preserved
	# Since it is also faster, it is better to use raw dict()
	OrderedDict = dict
else:
	try:
		# Python 2.7 and 3.1+
		from collections import OrderedDict
	except ImportError:
		# Tests will not work correctly but it doesn't affect the
		# functionality
		OrderedDict = dict

try:
	# Python 3.4+
	from statistics import median_low
except ImportError:

	def median_low(data):
		"""Returns the median of the input data, using the lower median for even-length data."""
		sorts = sorted(data)
		length = len(sorts)
		return sorts[(length - 1) // 2]


try:
	# Python 3.8+
	from math import isqrt
except ImportError:

	def isqrt(n):
		"""Compute the integer square root of a nonnegative integer."""
		# return int(math.sqrt(x))
		if n < 0:
			msg = "isqrt() argument must be nonnegative"
			raise ValueError(msg)
		if n == 0:
			return 0

		c = (n.bit_length() - 1) // 2
		a = 1
		d = 0
		for s in reversed(range(c.bit_length())):
			# Loop invariant: (a-1)**2 < (n >> 2*(c - d)) < (a+1)**2
			e = d
			d = c >> s
			a = (a << d - e - 1) + (n >> 2 * c - e - d + 1) // a

		return a - (a * a > n)


try:
	# Python 3.3+
	from math import log2
except ImportError:

	def log2(x):
		"""Calculate the base-2 logarithm of a given number."""
		return math.log(x, 2)


try:
	# Python 3.2+
	from math import expm1
except ImportError:

	def expm1(x):
		"""Return exp(x) - 1, the exponential of x minus 1."""
		return math.exp(x) - 1


if sys.platform == "win32":  # Windows
	from ctypes import wintypes

	try:
		# Python 3+
		import winreg
	except ImportError:
		import _winreg as winreg

	kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

	class GROUP_AFFINITY(ctypes.Structure):
		_fields_ = (("Mask", wintypes.WPARAM), ("Group", wintypes.WORD), ("Reserved", wintypes.WORD * 3))

	class PROCESSOR_RELATIONSHIP(ctypes.Structure):
		_fields_ = (
			("Flags", wintypes.BYTE),
			("EfficiencyClass", wintypes.BYTE),
			("Reserved", wintypes.BYTE * 20),
			("GroupCount", wintypes.WORD),
			("GroupMask", GROUP_AFFINITY * 1),
		)

	class union(ctypes.Union):
		_fields_ = (("GroupMask", GROUP_AFFINITY), ("GroupMasks", GROUP_AFFINITY * 1))

	class NUMA_NODE_RELATIONSHIP(ctypes.Structure):
		_fields_ = (
			("NodeNumber", wintypes.DWORD),
			("Reserved", wintypes.BYTE * 18),
			("GroupCount", wintypes.WORD),
			("union", union),
		)

		_anonymous_ = ("union",)

	class CACHE_RELATIONSHIP(ctypes.Structure):
		_fields_ = (
			("Level", wintypes.BYTE),
			("Associativity", wintypes.BYTE),
			("LineSize", wintypes.WORD),
			("CacheSize", wintypes.DWORD),
			("Type", wintypes.DWORD),
			("Reserved", wintypes.BYTE * 18),
			("GroupCount", wintypes.WORD),
			("union", union),
		)

		_anonymous_ = ("union",)

	class PROCESSOR_GROUP_INFO(ctypes.Structure):
		_fields_ = (
			("MaximumProcessorCount", wintypes.BYTE),
			("ActiveProcessorCount", wintypes.BYTE),
			("Reserved", wintypes.BYTE * 38),
			("ActiveProcessorMask", wintypes.WPARAM),
		)

	class GROUP_RELATIONSHIP(ctypes.Structure):
		_fields_ = (
			("MaximumGroupCount", wintypes.WORD),
			("ActiveGroupCount", wintypes.WORD),
			("Reserved", wintypes.BYTE * 20),
			("GroupInfo", PROCESSOR_GROUP_INFO * 1),
		)

	class union(ctypes.Union):
		_fields_ = (
			("Processor", PROCESSOR_RELATIONSHIP),
			("NumaNode", NUMA_NODE_RELATIONSHIP),
			("Cache", CACHE_RELATIONSHIP),
			("Group", GROUP_RELATIONSHIP),
		)

	class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
		_fields_ = (("Relationship", wintypes.DWORD), ("Size", wintypes.DWORD), ("union", union))

		_anonymous_ = ("union",)

	class ProcessorCore(ctypes.Structure):
		_fields_ = (("Flags", wintypes.BYTE),)

	class NumaNode(ctypes.Structure):
		_fields_ = (("NodeNumber", wintypes.DWORD),)

	class CACHE_DESCRIPTOR(ctypes.Structure):
		_fields_ = (
			("Level", wintypes.BYTE),
			("Associativity", wintypes.BYTE),
			("LineSize", wintypes.WORD),
			("Size", wintypes.DWORD),
			("Type", wintypes.DWORD),
		)

	class union(ctypes.Union):
		_fields_ = (
			("ProcessorCore", ProcessorCore),
			("NumaNode", NumaNode),
			("Cache", CACHE_DESCRIPTOR),
			("Reserved", ctypes.c_ulonglong * 2),
		)

	class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
		_fields_ = (("ProcessorMask", wintypes.WPARAM), ("Relationship", wintypes.DWORD), ("union", union))

		_anonymous_ = ("union",)

	class MEMORYSTATUSEX(ctypes.Structure):
		_fields_ = (
			("dwLength", wintypes.DWORD),
			("dwMemoryLoad", wintypes.DWORD),
			("ullTotalPhys", ctypes.c_ulonglong),
			("ullAvailPhys", ctypes.c_ulonglong),
			("ullTotalPageFile", ctypes.c_ulonglong),
			("ullAvailPageFile", ctypes.c_ulonglong),
			("ullTotalVirtual", ctypes.c_ulonglong),
			("ullAvailVirtual", ctypes.c_ulonglong),
			("ullAvailExtendedVirtual", ctypes.c_ulonglong),
		)

		def __init__(self):
			self.dwLength = ctypes.sizeof(self)
			super(MEMORYSTATUSEX, self).__init__()

elif sys.platform == "darwin":  # macOS
	libc = ctypes.CDLL(find_library("c"))

	def sysctl_str(name):
		size = ctypes.c_size_t()
		libc.sysctlbyname(name, None, ctypes.byref(size), None, 0)

		buf = ctypes.create_string_buffer(size.value)
		libc.sysctlbyname(name, buf, ctypes.byref(size), None, 0)
		return buf.value

	def sysctl_value(name, ctype):
		size = ctypes.c_size_t(ctypes.sizeof(ctype))
		value = ctype()
		libc.sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0)
		return value.value

elif sys.platform.startswith("linux"):
	try:
		# Python 3.10+
		from platform import freedesktop_os_release
	except ImportError:

		def freedesktop_os_release():
			line_re = re.compile(r"""^([a-zA-Z_][a-zA-Z0-9_]*)=('([^']*)'|"((?:[^$"`]|\\[$"`\\])*)"|.*)$""")
			quote_unescape_re = re.compile(r'\\([$"`\\])')
			unescape_re = re.compile(r"\\(.)")
			info = {}
			for candidate in ("/etc/os-release", "/usr/lib/os-release"):
				if os.path.isfile(candidate):
					with io.open(candidate, encoding="utf-8") as file:
						# lexer = shlex.shlex(file, posix=True)
						# lexer.whitespace_split = True
						for line in file:
							if not line or line.startswith("#"):
								continue
							match = line_re.match(line)
							if match:
								info[match.group(1)] = (
									match.group(3)
									if match.group(3) is not None
									else quote_unescape_re.sub(r"\1", match.group(4))
									if match.group(4) is not None
									else unescape_re.sub(r"\1", match.group(2))
								)
					break
			return info

	libc = ctypes.CDLL(find_library("c"))

	class sysinfo(ctypes.Structure):
		_fields_ = (
			("uptime", ctypes.c_long),
			("loads", ctypes.c_ulong * 3),
			("totalram", ctypes.c_ulong),
			("freeram", ctypes.c_ulong),
			("sharedram", ctypes.c_ulong),
			("bufferram", ctypes.c_ulong),
			("totalswap", ctypes.c_ulong),
			("freeswap", ctypes.c_ulong),
			("procs", ctypes.c_ushort),
			("pad", ctypes.c_ushort),
			("totalhigh", ctypes.c_ulong),
			("freehigh", ctypes.c_ulong),
			("mem_unit", ctypes.c_uint),
			("_f", ctypes.c_char * (20 - 2 * ctypes.sizeof(ctypes.c_int) - ctypes.sizeof(ctypes.c_long))),
		)


cl_lib = find_library("OpenCL")
if cl_lib:
	cl = ctypes.CDLL(cl_lib)

nvml_lib = find_library("nvml" if sys.platform == "win32" else "nvidia-ml")
if nvml_lib:
	nvml = ctypes.CDLL(nvml_lib)


class nvmlMemory_t(ctypes.Structure):
	_fields_ = (("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong))


try:
	# Python 3.3+
	from shutil import disk_usage
except ImportError:
	# Adapted from: https://code.activestate.com/recipes/577972-disk-usage/
	from collections import namedtuple

	_ntuple_diskusage = namedtuple("usage", "total used free")

	if hasattr(os, "statvfs"):  # POSIX

		def disk_usage(path):
			st = os.statvfs(path)
			free = st.f_bavail * st.f_frsize
			total = st.f_blocks * st.f_frsize
			used = (st.f_blocks - st.f_bfree) * st.f_frsize
			return _ntuple_diskusage(total, used, free)

	elif os.name == "nt":  # Windows
		ctypes.windll.kernel32.GetDiskFreeSpaceExW.argtypes = (
			wintypes.LPCWSTR,
			ctypes.POINTER(ctypes.c_ulonglong),
			ctypes.POINTER(ctypes.c_ulonglong),
			ctypes.POINTER(ctypes.c_ulonglong),
		)
		ctypes.windll.kernel32.GetDiskFreeSpaceExW.restype = wintypes.BOOL

		ctypes.windll.kernel32.GetDiskFreeSpaceExA.argtypes = (
			wintypes.LPCSTR,
			ctypes.POINTER(ctypes.c_ulonglong),
			ctypes.POINTER(ctypes.c_ulonglong),
			ctypes.POINTER(ctypes.c_ulonglong),
		)
		ctypes.windll.kernel32.GetDiskFreeSpaceExA.restype = wintypes.BOOL

		def disk_usage(path):
			_, total, free = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
			fun = (
				ctypes.windll.kernel32.GetDiskFreeSpaceExW
				if sys.version_info >= (3,) or isinstance(path, str)
				else ctypes.windll.kernel32.GetDiskFreeSpaceExA
			)
			if not fun(path, ctypes.byref(_), ctypes.byref(total), ctypes.byref(free)):
				raise ctypes.WinError()
			used = total.value - free.value
			return _ntuple_diskusage(total.value, used, free.value)


try:
	# Python 3.3+
	from os import replace
except ImportError:
	if os.name == "nt":  # Windows
		ctypes.windll.kernel32.MoveFileExW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
		ctypes.windll.kernel32.MoveFileExW.restype = wintypes.BOOL

		ctypes.windll.kernel32.MoveFileExA.argtypes = (wintypes.LPCSTR, wintypes.LPCSTR, wintypes.DWORD)
		ctypes.windll.kernel32.MoveFileExA.restype = wintypes.BOOL

		def replace(src, dst):
			fun = (
				ctypes.windll.kernel32.MoveFileExW
				if sys.version_info >= (3,) or isinstance(src, str) or isinstance(dst, str)
				else ctypes.windll.kernel32.MoveFileExA
			)
			if not fun(src, dst, 0x1):  # MOVEFILE_REPLACE_EXISTING
				raise ctypes.WinError()

	else:  # POSIX
		replace = os.rename

try:
	# Windows
	import winsound
except ImportError:

	def beep():
		"""Emits a beep sound."""
		print("\a")

else:

	def beep():
		"""Plays a default system notification sound."""
		winsound.MessageBeep(type=-1)


try:
	# Linux and macOS
	import readline
except ImportError:
	pass
else:
	readline.set_completer_delims("")
	readline.parse_and_bind("tab: complete")

try:
	# Python 3.5+
	from json.decoder import JSONDecodeError
except ImportError:
	JSONDecodeError = ValueError

try:
	# import certifi
	import requests
	import urllib3
	from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout
except ImportError:
	executable = os.path.basename(sys.executable) if sys.executable else "python3"
	print(
		"""Please run the below command to install the Requests library:

	{0} -m pip install requests

Then, run AutoPrimeNet again.""".format(executable[:-4] if executable.endswith(".exe") else executable)
	)
	sys.exit(0)

locale.setlocale(locale.LC_ALL, "")
if hasattr(sys, "set_int_max_str_digits"):
	sys.set_int_max_str_digits(0)
charset.add_charset("utf-8", charset.QP, charset.QP, "utf-8")

VERSION = "1.0.2"
# GIMPS programs to use in the application version string when registering with PrimeNet
PROGRAMS = (
	{"name": "Prime95", "version": "30.19", "build": 20},
	{"name": "Mlucas", "version": "21.0.1"},
	{"name": "GpuOwl", "version": "7.5"},
	{"name": "CUDALucas", "version": "2.06"},
	{"name": "mfaktc", "version": "0.23"},
	{"name": "mfakto", "version": "0.16"},
)
# People to e-mail when a new prime is found
# E-mail addresses munged to prevent spam
CCEMAILS = [
	(name, user + "@" + ".".join(hosts))
	for name, (user, hosts) in (
		# ("Primenet server", ("primenet", ("mersenne", "org"))),
		("George Woltman", ("woltman", ("alum", "mit", "edu"))),
		("James Heinrich", ("james", ("mersenne", "ca"))),
		("Aaron", ("aaron", ("madpoo", "com"))),
		# ("E. Mayer", ("ewmayer", ("aol", "com"))),
		("Mihai Preda", ("mpreda", ("gmail", "com"))),
		("Daniel Connelly", ("connellyd2050", ("gmail", "com"))),
		("Teal Dulcet", ("tdulcet", ("gmail", "com"))),
	)
]

primenet_v5_burl = "http://v5.mersenne.org/v5server/"
TRANSACTION_API_VERSION = 0.95
ERROR_RATE = 0.018  # Estimated LL error rate on clean run
# Estimated PRP error rate (assumes Gerbicz error-checking)
PRP_ERROR_RATE = 0.0001
_V5_UNIQUE_TRUSTED_CLIENT_CONSTANT_ = 17737
primenet_v5_bargs = OrderedDict((("px", "GIMPS"), ("v", TRANSACTION_API_VERSION)))
primenet_baseurl = "https://www.mersenne.org/"
mersenne_ca_baseurl = "https://www.mersenne.ca/"
MAX_PRIMENET_EXP = 1000000000

is_64bit = platform.machine().endswith("64")
PORT = None
if sys.platform == "win32":
	PORT = 4 if is_64bit else 1
elif sys.platform == "darwin":
	PORT = 10 if is_64bit else 9
elif sys.platform.startswith("linux"):
	PORT = 8 if is_64bit else 2

session = requests.Session()  # session that maintains our cookies
session.headers["User-Agent"] = "AutoPrimeNet assignment handler version {0} ({1} {2}/{3})".format(
	VERSION, session.headers["User-Agent"], platform.python_implementation(), platform.python_version()
)
# urllib3 1.26+: allowed_methods=None, method_whitelist=None
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=urllib3.util.Retry(3, backoff_factor=1)))
session.mount("http://", requests.adapters.HTTPAdapter(max_retries=urllib3.util.Retry(3, backoff_factor=1)))
atexit.register(session.close)
# Python 2.7.9 and 3.4+
# context = ssl.create_default_context(cafile=certifi.where())

# Mlucas constants

TEST_TYPE_PRIMALITY = 1
TEST_TYPE_PRP = 2
TEST_TYPE_PM1 = 3

MODULUS_TYPE_MERSENNE = 1
MODULUS_TYPE_FERMAT = 3


class timedelta(timedelta):
	"""Custom timedelta class with a formatted string representation."""

	def __str__(self):
		"""Return a formatted string representation of the timedelta."""
		m, s = divmod(self.seconds, 60)
		h, m = divmod(m, 60)
		d = self.days
		# self.microseconds
		return "{0}{1}{2}{3}".format(
			"{0:n}d".format(d) if d else "",
			"{0:02n}h".format(h) if d else "{0:n}h".format(h) if h else "",
			"{0:02n}m".format(m) if h or d else "{0:n}m".format(m) if m else "",
			"{0:02n}s".format(s) if m or h or d else "{0:n}s".format(s),
		)


class Formatter(logging.Formatter):
	"""Custom logging formatter to include worker information if available."""

	def format(self, record):
		"""Format log record to include worker number if 'cpu_num' attribute is present."""
		record.worker = ", Worker #{0:n}".format(record.cpu_num + 1) if hasattr(record, "cpu_num") else ""
		return super(Formatter, self).format(record)


class COLORS:
	"""ANSI escape sequences for terminal text colors."""

	RED = "\033[31m"
	GREEN = "\033[32m"
	YELLOW = "\033[33m"
	# BLUE = "\033[34m"
	MAGENTA = "\033[35m"
	# CYAN = "\033[36m"
	GRAY = "\033[90m"
	DEFAULT = "\033[39m"  # Default Color


BOLD = "\033[1m"
# DIM = "\033[2m"
# RESET = "\033[22m"
RESET_All = "\033[m"


class ColorFormatter(Formatter):
	"""Custom log formatter to add color based on log level."""

	FORMATS = {
		logging.DEBUG: COLORS.GRAY,
		# logging.INFO: COLORS.GREEN,
		logging.WARNING: COLORS.YELLOW,
		logging.ERROR: COLORS.RED,
		logging.CRITICAL: COLORS.MAGENTA,
	}

	def format(self, record):
		"""Format log record with color based on log level."""
		fmt = super(ColorFormatter, self).format(record)
		color = self.FORMATS.get(record.levelno)
		if COLOR and color:
			return color + fmt + COLORS.DEFAULT
		return fmt


class LockFile:
	"""Context manager for creating and managing a lock file."""

	__slots__ = ("filename", "lockfile")

	def __init__(self, filename):
		"""Initialize with the name of the file to lock."""
		self.filename = filename
		self.lockfile = filename + ".lck"

	def __enter__(self):
		"""Acquire the lock by creating the lock file."""
		for i in count():
			try:
				# Python 3.3+: with open(self.lockfile, "x") as f:
				fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL)
				os.close(fd)
				break
			# Python 3.3+: FileExistsError
			except (IOError, OSError) as e:
				if e.errno == errno.EEXIST:
					if not i:
						logging.debug("%r lockfile already exists, waiting...", self.lockfile)
					time.sleep(min(1 << i, 60 * 1000) / 1000)
				else:
					logging.exception("Error opening %r lockfile: %s", self.lockfile, e, exc_info=options.debug)
					raise
		if i:
			logging.debug("Locked %r", self.filename)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Release the lock by removing the lock file."""
		os.remove(self.lockfile)


class SEC:
	Internals = "Internals"
	PrimeNet = "PrimeNet"
	Email = "Email"


class PRIMENET:
	# Error codes returned to client
	ERROR_OK = 0  # no error
	ERROR_SERVER_BUSY = 3  # server is too busy now
	ERROR_INVALID_VERSION = 4
	ERROR_INVALID_TRANSACTION = 5
	# Returned for length, type, or character invalidations.
	ERROR_INVALID_PARAMETER = 7
	ERROR_ACCESS_DENIED = 9
	ERROR_DATABASE_CORRUPT = 11
	ERROR_DATABASE_FULL_OR_BROKEN = 13
	# Account related errors:
	ERROR_INVALID_USER = 21
	# Computer cpu/software info related errors:
	ERROR_UNREGISTERED_CPU = 30
	ERROR_OBSOLETE_CLIENT = 31
	ERROR_STALE_CPU_INFO = 32
	ERROR_CPU_IDENTITY_MISMATCH = 33
	ERROR_CPU_CONFIGURATION_MISMATCH = 34
	# Work assignment related errors:
	ERROR_NO_ASSIGNMENT = 40
	ERROR_INVALID_ASSIGNMENT_KEY = 43
	ERROR_INVALID_ASSIGNMENT_TYPE = 44
	ERROR_INVALID_RESULT_TYPE = 45
	ERROR_INVALID_WORK_TYPE = 46
	ERROR_WORK_NO_LONGER_NEEDED = 47
	# Undocumented and missing from Prime95/MPrime
	ERROR_ILLEGAL_RESIDUE = 48

	# Valid work_preference values
	WP_WHATEVER = 0  # Whatever makes most sense
	WP_FACTOR_LMH = 1  # Factor big numbers to low limits
	WP_FACTOR = 2  # Trial factoring
	WP_PMINUS1 = 3  # P-1 of small Mersennes --- not supported
	WP_PFACTOR = 4  # P-1 of large Mersennes
	WP_ECM_SMALL = 5  # ECM of small Mersennes looking for first factors
	WP_ECM_FERMAT = 6  # ECM of Fermat numbers
	WP_ECM_CUNNINGHAM = 7  # ECM of Cunningham numbers --- not supported
	WP_ECM_COFACTOR = 8  # ECM of Mersenne cofactors
	WP_GPU_FACTOR = 12  # Trial Factoring on Mersennes at wavefront exponents
	WP_LL_FIRST = 100  # LL first time tests
	WP_LL_DBLCHK = 101  # LL double checks
	WP_LL_WORLD_RECORD = 102  # LL test of world record Mersenne
	WP_LL_100M = 104  # LL 100 million digit
	WP_PRP_FIRST = 150  # PRP test of big Mersennes
	WP_PRP_DBLCHK = 151  # PRP double checks
	WP_PRP_WORLD_RECORD = 152  # PRP test of world record Mersennes
	WP_PRP_100M = 153  # PRP test of 100M digit Mersennes
	WP_PRP_NO_PMINUS1 = 154  # PRP test that if possible also needs P-1
	WP_PRP_DC_PROOF = 155  # PRP double-check where a proof will be produced
	WP_PRP_COFACTOR = 160  # PRP test of Mersenne cofactors
	WP_PRP_COFACTOR_DBLCHK = 161  # PRP double check of Mersenne cofactors

	# Valid work_types returned by ga
	WORK_TYPE_FACTOR = 2
	WORK_TYPE_PMINUS1 = 3
	WORK_TYPE_PFACTOR = 4
	WORK_TYPE_ECM = 5
	WORK_TYPE_PPLUS1 = 6  # Not yet supported by the server
	WORK_TYPE_FIRST_LL = 100
	WORK_TYPE_DBLCHK = 101
	WORK_TYPE_PRP = 150
	WORK_TYPE_CERT = 200

	# This structure is passed for the ar - Assignment Result call
	AR_NO_RESULT = 0  # No result, just sending done msg
	AR_TF_FACTOR = 1  # Trial factoring, factor found
	AR_P1_FACTOR = 2  # P-1, factor found
	AR_ECM_FACTOR = 3  # ECM, factor found
	AR_TF_NOFACTOR = 4  # Trial Factoring no factor found
	AR_P1_NOFACTOR = 5  # P-1 factoring no factor found
	AR_ECM_NOFACTOR = 6  # ECM factoring no factor found
	AR_PP1_FACTOR = 7  # P+1, factor found
	AR_PP1_NOFACTOR = 8  # P+1 factoring no factor found
	AR_LL_RESULT = 100  # LL result, not prime
	AR_LL_PRIME = 101  # LL result, Mersenne prime
	AR_PRP_RESULT = 150  # PRP result, not prime
	AR_PRP_PRIME = 151  # PRP result, probably prime
	AR_CERT = 200  # Certification result


ERRORS = {
	PRIMENET.ERROR_SERVER_BUSY: "Server busy",
	PRIMENET.ERROR_INVALID_VERSION: "Invalid version",
	PRIMENET.ERROR_INVALID_TRANSACTION: "Invalid transaction",
	PRIMENET.ERROR_INVALID_PARAMETER: "Invalid parameter",
	PRIMENET.ERROR_ACCESS_DENIED: "Access denied",
	PRIMENET.ERROR_DATABASE_CORRUPT: "Server database malfunction",
	PRIMENET.ERROR_DATABASE_FULL_OR_BROKEN: "Server database full or broken",
	PRIMENET.ERROR_INVALID_USER: "Invalid user",
	PRIMENET.ERROR_UNREGISTERED_CPU: "CPU not registered",
	PRIMENET.ERROR_OBSOLETE_CLIENT: "Obsolete client, please upgrade",
	PRIMENET.ERROR_STALE_CPU_INFO: "Stale cpu info",
	PRIMENET.ERROR_CPU_IDENTITY_MISMATCH: "CPU identity mismatch",
	PRIMENET.ERROR_CPU_CONFIGURATION_MISMATCH: "CPU configuration mismatch",
	PRIMENET.ERROR_NO_ASSIGNMENT: "No assignment",
	PRIMENET.ERROR_INVALID_ASSIGNMENT_KEY: "Invalid assignment key",
	PRIMENET.ERROR_INVALID_ASSIGNMENT_TYPE: "Invalid assignment type",
	PRIMENET.ERROR_INVALID_RESULT_TYPE: "Invalid result type",
	# Missing from Prime95/MPrime
	PRIMENET.ERROR_INVALID_WORK_TYPE: "Invalid work type",
	PRIMENET.ERROR_WORK_NO_LONGER_NEEDED: "Work no longer needed",
	# Undocumented
	PRIMENET.ERROR_ILLEGAL_RESIDUE: "Illegal residue",
}


class Assignment(object):
	"""Assignment(work_type, uid, k, b, n, c, sieve_depth, factor_to, pminus1ed, B1, B2, B2_start, tests_saved, prp_base, prp_residue_type, prp_dblchk, known_factors, ra_failed, cert_squarings)."""

	__slots__ = (
		"work_type",
		"uid",
		"k",
		"b",
		"n",
		"c",
		"sieve_depth",
		"factor_to",
		"pminus1ed",
		"B1",
		"B2",
		"B2_start",
		"tests_saved",
		"prp_base",
		"prp_residue_type",
		"prp_dblchk",
		"known_factors",
		"ra_failed",
		"cert_squarings",
	)

	def __init__(self, work_type=None):
		"""Create new instance of Assignment(work_type, uid, k, b, n, c, sieve_depth, factor_to, pminus1ed, B1, B2, B2_start, tests_saved, prp_base, prp_residue_type, prp_dblchk, known_factors, ra_failed, cert_squarings)."""
		self.work_type = work_type
		self.uid = None
		# k*b^n+c
		self.k = 1.0
		self.b = 2
		self.n = 0
		self.c = -1
		self.sieve_depth = 99.0
		self.factor_to = 0.0
		self.pminus1ed = 1
		self.B1 = 0
		self.B2 = 0
		self.B2_start = 0
		self.tests_saved = 0.0
		self.prp_base = 0
		self.prp_residue_type = 0
		self.prp_dblchk = False
		self.known_factors = None
		self.ra_failed = False
		self.cert_squarings = 0


suffix_power_char = ("", "K", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q")
suffix_power = {"k": 1, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5, "E": 6, "Z": 7, "Y": 8, "R": 9, "Q": 10}

# YES_RE = re.compile(locale.nl_langinfo(locale.YESEXPR))
YES_RE = re.compile(r"^[yY]")
# NO_RE = re.compile(locale.nl_langinfo(locale.NOEXPR))
NO_RE = re.compile(r"^[nN]")

# Python 2
if hasattr(__builtins__, "unicode"):
	str = unicode

# Python 2
if hasattr(__builtins__, "raw_input"):
	input = raw_input

# Python 2
if hasattr(__builtins__, "xrange"):
	range = xrange


def exponent_to_str(assignment):
	"""Converts an assignment's exponent to a formatted string representation."""
	if not assignment.n:
		buf = "{0:.0f}".format(assignment.k + assignment.c)
	elif assignment.k != 1.0:
		buf = "{0.k:.0f}*{0.b}^{0.n}{0.c:+}".format(assignment)
	elif assignment.b == 2 and assignment.c == -1:
		buf = "M{0.n}".format(assignment)
		if assignment.work_type == PRIMENET.WORK_TYPE_FACTOR:
			buf += " (TF:{0.sieve_depth:.0f}-{0.factor_to:.0f})".format(assignment)
	else:
		cnt = 0
		temp_n = assignment.n
		while not temp_n & 1:
			temp_n >>= 1
			cnt += 1
		if assignment.b == 2 and temp_n == 1 and assignment.c == 1:
			buf = "F{0}".format(cnt)
		else:
			buf = "{0.b}^{0.n}{0.c:+}".format(assignment)
	return buf


def exponent_to_text(assignment):
	"""Converts an assignment's work type and exponent to a descriptive text string."""
	if assignment.work_type == PRIMENET.WORK_TYPE_FIRST_LL:
		work_type_str = "LL"
	elif assignment.work_type == PRIMENET.WORK_TYPE_DBLCHK:
		work_type_str = "Double check"
	elif assignment.work_type == PRIMENET.WORK_TYPE_PRP:
		work_type_str = "PRPDC" if assignment.prp_dblchk else "PRP"
	elif assignment.work_type == PRIMENET.WORK_TYPE_FACTOR:
		work_type_str = "Trial factor"
	elif assignment.work_type in {PRIMENET.WORK_TYPE_PFACTOR, PRIMENET.WORK_TYPE_PMINUS1}:
		work_type_str = "P-1"
	elif assignment.work_type == PRIMENET.WORK_TYPE_CERT:
		work_type_str = "CERT"
	return "{0} {1}".format(work_type_str, exponent_to_str(assignment))


def assignment_to_str(assignment):
	"""Converts an assignment object to its string representation, including known factors if present."""
	buf = exponent_to_str(assignment)
	if not assignment.known_factors:
		return buf
	return "{0}/{1}".format("({0})".format(buf) if "^" in buf else buf, "/".join(map(str, assignment.known_factors)))


def outputunit(number, scale=False):
	"""Converts a number to a human-readable string with appropriate scaling and suffix."""
	scale_base = 1000 if scale else 1024

	power = 0
	while abs(number) >= scale_base:
		power += 1
		number /= scale_base

	anumber = abs(number)
	anumber += 0.0005 if anumber < 10 else 0.005 if anumber < 100 else 0.05 if anumber < 1000 else 0.5

	if number and anumber < 1000 and power > 0:
		strm = "{0:.{prec}g}".format(number, prec=sys.float_info.dig)

		length = 5 + (number < 0)
		if len(strm) > length:
			prec = 3 if anumber < 10 else 2 if anumber < 100 else 1
			strm = "{0:.{prec}f}".format(number, prec=prec)
	else:
		strm = "{0:.0f}".format(number)

	# "k" if power == 1 and scale else
	strm += " " + (suffix_power_char[power] if power < len(suffix_power_char) else "(error)")

	if not scale and power > 0:
		strm += "i"

	return strm


def inputunit(astr, scale=False):
	"""Converts a string with a unit suffix to an integer value."""
	scale_base = 1000 if scale else 1024

	unit = astr[-1]
	if unit in suffix_power:
		return int(float(astr[:-1]) * scale_base ** suffix_power[unit])

	return int(astr)


def output_available(available, total):
	"""Formats the available and total byte values into a human-readable string."""
	return "{0}B / {1}B{2}".format(
		outputunit(available),
		outputunit(total),
		" ({0}B / {1}B)".format(outputunit(available, True), outputunit(total, True)) if total >= 1000 else "",
	)


def ask_yn(astr, val):
	"""Prompt the user with a yes/no question and return the response as a boolean."""
	while True:
		temp = input("{0} ({1}): ".format(astr, "Y" if val else "N")).strip()
		if not temp:
			return val
		yes_res = YES_RE.match(temp)
		no_res = NO_RE.match(temp)
		if yes_res or no_res:
			return bool(yes_res)


def ask_int(astr, val, amin=None, amax=None, base=0):
	"""Prompt the user for an integer input with optional default value, range, and base."""
	while True:
		temp = input("{0}{1}: ".format(astr, " ({0!r})".format(val) if val is not None else ""))
		if not temp:
			return val
		try:
			newval = int(temp, base)
		except ValueError:
			print("Please enter a valid number.")
			continue
		if (amin is None or newval >= amin) and (amax is None or newval <= amax):
			return newval
		if amin is not None and amax is not None:
			print("Please enter a number between {0:n} and {1:n}.".format(amin, amax))
		elif amin is not None:
			print("Please enter a number greater than or equal to {0:n}.".format(amin))
		elif amax is not None:
			print("Please enter a number less than or equal to {0:n}.".format(amax))


def ask_float(astr, val, amin=None, amax=None):
	"""Prompt the user for a float input with optional default value and range."""
	while True:
		temp = input("{0}{1}: ".format(astr, " ({0!r})".format(val) if val is not None else ""))
		if not temp:
			return val
		try:
			newval = float(temp)
		except ValueError:
			print("Please enter a valid number.")
			continue
		if (amin is None or newval >= amin) and (amax is None or newval <= amax):
			return newval
		if amin is not None and amax is not None:
			print("Please enter a number between {0:n} and {1:n}.".format(amin, amax))
		elif amin is not None:
			print("Please enter a number greater than or equal to {0:n}.".format(amin))
		elif amax is not None:
			print("Please enter a number less than or equal to {0:n}.".format(amax))


def ask_str(astr, val, maxlen=0):
	"""Prompt the user for a string input with optional default value and maximum length."""
	while True:
		temp = input("{0}{1}: ".format(astr, " ({0!r})".format(val) if val else "")).strip()
		if not temp:
			return val
		if not maxlen or len(temp) <= maxlen:
			return temp
		print("Maximum string length is {0:n} characters.".format(maxlen))
		val = temp[:maxlen]


def ask_pass(astr, val):
	"""Prompt the user for a password, displaying asterisks for existing input if provided."""
	return getpass.getpass("{0}{1}: ".format(astr, " ({0})".format("*" * len(val)) if val else "")) or val


def ask_ok():
	"""Prompts the user to hit Enter to continue."""
	input("\nHit Enter to continue: ")


def ask_ok_cancel():
	"""Prompt the user with a yes/no question to accept the answers above."""
	return ask_yn("\nAccept the answers above?", True)


def get_device_str(device, name):
	"""Retrieve the specified information string from an OpenCL device."""
	size = ctypes.c_size_t()
	cl.clGetDeviceInfo(device, name, ctypes.c_size_t(0), None, ctypes.byref(size))

	buf = ctypes.create_string_buffer(size.value)
	cl.clGetDeviceInfo(device, name, size, buf, None)
	return buf.value


def get_device_value(device, name, ctype):
	"""Retrieve the specified information value from an OpenCL device."""
	size = ctypes.sizeof(ctype)
	value = ctype()
	cl.clGetDeviceInfo(device, name, size, ctypes.byref(value), None)
	return value.value


def get_opencl_devices():
	"""Retrieves a list of available OpenCL devices with their names, maximum clock frequencies, and global memory sizes."""
	num_platforms = ctypes.c_uint()
	result = cl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
	if result and result != -1001:  # CL_PLATFORM_NOT_FOUND_KHR
		logging.error("Failed to get the number of OpenCL platforms: %s", result)
		return []
	logging.debug("Number of platforms: %s", num_platforms.value)
	if not num_platforms.value:
		return []

	platforms = (ctypes.c_void_p * num_platforms.value)()
	if cl.clGetPlatformIDs(num_platforms.value, platforms, None):
		logging.error("Failed to get the OpenCL platforms")
		return []

	adevices = []

	for aplatform in map(ctypes.c_void_p, platforms):
		num_devices = ctypes.c_uint()
		result = cl.clGetDeviceIDs(aplatform, 0xFFFFFFFF, 0, None, ctypes.byref(num_devices))  # CL_DEVICE_TYPE_ALL
		if result and result != -1:  # CL_DEVICE_NOT_FOUND
			logging.error("Failed to get the number of OpenCL devices: %s", result)
			continue
		logging.debug("\tNumber of devices: %s", num_devices.value)
		if not num_devices.value:
			continue

		devices = (ctypes.c_void_p * num_devices.value)()
		if cl.clGetDeviceIDs(aplatform, 0xFFFFFFFF, num_devices.value, devices, None):  # CL_DEVICE_TYPE_ALL
			logging.error("Failed to get the OpenCL devices")
			continue
		for device in map(ctypes.c_void_p, devices):
			name = get_device_str(device, 0x102B)  # CL_DEVICE_NAME

			freq = get_device_value(device, 0x100C, ctypes.c_uint)  # CL_DEVICE_MAX_CLOCK_FREQUENCY

			mem = get_device_value(device, 0x101F, ctypes.c_uint64)  # CL_DEVICE_GLOBAL_MEM_SIZE
			memory = mem >> 20

			adevices.append((name.decode("utf-8"), freq, memory))

	return adevices


def get_nvidia_devices():
	"""Retrieve a list of Nvidia GPU devices with their names, maximum clock frequencies, and total memory."""
	if nvml.nvmlInit():
		logging.error("Failed to initialize NVML")
		return []
	try:
		device_count = ctypes.c_uint()
		if nvml.nvmlDeviceGetCount(ctypes.byref(device_count)):
			logging.error("Failed to get the Nvidia device count")
			return []
		logging.debug("Total Devices: %s", device_count.value)

		devices = []

		for i in range(device_count.value):
			device = ctypes.c_void_p()
			if nvml.nvmlDeviceGetHandleByIndex(i, ctypes.byref(device)):
				# raise Exception
				logging.error("Failed to get handle for Nvidia device %s", i)
				continue

			buf = ctypes.create_string_buffer(96)  # NVML_DEVICE_NAME_V2_BUFFER_SIZE
			nvml.nvmlDeviceGetName(device, buf, len(buf))
			name = buf.value

			clock = ctypes.c_uint()
			nvml.nvmlDeviceGetMaxClockInfo(device, 0, ctypes.byref(clock))
			freq = clock.value

			mem = nvmlMemory_t()
			nvml.nvmlDeviceGetMemoryInfo(device, ctypes.byref(mem))
			memory = mem.total >> 20

			devices.append((name.decode("utf-8"), freq, memory))
	finally:
		nvml.nvmlShutdown()

	return devices


def get_gpus():
	"""Retrieve a list of available GPU devices from OpenCL and Nvidia libraries."""
	gpus = []

	if cl_lib:
		logging.debug("OpenCL")
		gpus.extend(get_opencl_devices())
	else:
		logging.debug("OpenCL library not found on this system.")

	if nvml_lib:
		logging.debug("Nvidia")
		gpus.extend(get_nvidia_devices())
	else:
		logging.debug("NVML library not found on this system.")

	return gpus


def setup():
	"""Configures the GIMPS/PrimeNet client with user preferences and system settings."""
	wrapper = textwrap.TextWrapper(width=75)
	print(
		wrapper.fill(
			"Welcome to GIMPS, the hunt for huge prime numbers.  The program will ask you a few simple questions and then contact the PrimeNet server to get some work for your computer.  Good luck!"
		)
	)

	print(
		"\n"
		+ wrapper.fill(
			"Create a GIMPS/PrimeNet account: https://www.mersenne.org/update/ or you may contribute anonymously but it is not recommended."
		)
	)
	userid = ask_str('Your GIMPS/PrimeNet user ID or "ANONYMOUS"', options.user_id or "ANONYMOUS", 20)
	compid = ask_str("Optional computer name", options.computer_id, 20)
	if ask_yn("Use a proxy server", False):
		print(
			"Use of a proxy is supported, but not (yet) configurable. Please let us know that you need this feature and we can make it configurable."
		)

	if not ask_ok_cancel():
		return None
	if options.user_id != userid:
		options.user_id = userid
		config.set(SEC.PrimeNet, "username", userid)

	if options.computer_id != compid:
		options.computer_id = compid
		config.set(SEC.PrimeNet, "ComputerID", compid)

	program = ask_int(
		"Which GIMPS program are you getting assignments for (1=Mlucas, 2=GpuOwl, 3=PRPLL, 4=CUDALucas, 5=mfaktc, 6=mfakto)",
		6 if options.mfakto else 5 if options.mfaktc else 4 if options.cudalucas else 2 if options.gpuowl else 1,
		1,
		6,
	)
	if program == 3:
		print(
			"Unfortunately, PRPLL is not PrimeNet server compatible and is thus not (yet) fully supported. Please use GpuOwl instead."
		)
		sys.exit(1)
	tf1g = False
	if program in {5, 6}:
		tf1g = ask_yn(
			"Is this setup for the TF1G subproject on https://mersenne.ca/tf1G? This is mutually exclusive with PrimeNet trial factoring.",
			options.min_exp and options.min_exp >= MAX_PRIMENET_EXP,
		)
	gpu = None
	if program != 1:
		print("\nThis program can optionally report the Graphics Processor (GPU) to PrimeNet instead of the CPU")
		gpus = get_gpus()
		if gpus:
			print("Detected GPUs (some may be repeated):")
			for i, (name, freq, memory) in enumerate(gpus):
				print("\n{0:n}. {1}".format(i + 1, name))
				print("\tFrequency/Speed: {0:n} MHz".format(freq))
				print("\tTotal Memory: {0:n} MiB ({1}B)".format(memory, outputunit(memory << 20)))
			print()
			gpu = ask_int("Which GPU are you using this GIMPS program with (0 to not report the GPU)", 0, 0, len(gpus))
		else:
			print("No GPUs were detected\n")
	hours = ask_int("Hours per day you expect the GIMPS program will run", options.cpu_hours, 1, 24)

	if not ask_ok_cancel():
		return None
	if options.cpu_hours != hours:
		options.cpu_hours = hours
		config.set(SEC.PrimeNet, "CPUHours", str(hours))
		config.set(SEC.PrimeNet, "RollingAverage", str(1000))
	if program == 1:
		options.mlucas = True
		config.set(SEC.PrimeNet, "mlucas", str(True))
	elif program == 2:
		options.gpuowl = True
		config.set(SEC.PrimeNet, "gpuowl", str(True))
	elif program == 4:
		options.cudalucas = True
		config.set(SEC.PrimeNet, "cudalucas", str(True))
	elif program == 5:
		options.mfaktc = True
		config.set(SEC.PrimeNet, "mfaktc", str(True))
	elif program == 6:
		options.mfakto = True
		config.set(SEC.PrimeNet, "mfakto", str(True))
	if gpu:
		name, freq, memory = gpus[gpu - 1]
		options.cpu_brand = name
		config.set(SEC.PrimeNet, "CpuBrand", name)
		options.cpu_speed = freq
		config.set(SEC.PrimeNet, "CpuSpeed", str(freq))
		options.memory = memory
		config.set(SEC.PrimeNet, "memory", str(memory))
		options.day_night_memory = memory * 0.9
	if tf1g:
		options.min_exp = MAX_PRIMENET_EXP
		config.set(SEC.PrimeNet, "GetMinExponent", str(options.min_exp))
	elif options.min_exp and options.min_exp >= MAX_PRIMENET_EXP:
		options.min_exp = None
		config.remove_option(SEC.PrimeNet, "GetMinExponent")

	disk = ask_float(
		"Configured disk space limit per worker to store the proof interim residues files for PRP tests in GiB/worker (0 to not send)",
		options.worker_disk_space,
		0,
	)
	day_night_memory = ask_float("Configured day/night P-1 stage 2 memory in GiB", options.day_night_memory / 1024, 0)
	archive_dir = ask_str("Optional directory to archive PRP proof files after upload", options.archive_dir or "")
	# cert_cpu = ask_int("PRP proof certification work limit in percentage of CPU or GPU time", options.cert_cpu_limit, 1, 100)

	if not ask_ok_cancel():
		return None
	athreshold = 1.5
	threshold = athreshold * 1024**3
	if (not options.worker_disk_space or options.worker_disk_space >= athreshold) and disk and disk < athreshold:
		print(
			wrapper.fill(
				"Setting disk space limit below {0}B ({1}B) may preclude getting first time prime tests from the PrimeNet server. Consider setting it to 0 instead to not send.".format(
					outputunit(threshold), outputunit(threshold, True)
				)
			)
		)
		ask_ok()
	# options.worker_disk_space = disk
	config.set(SEC.PrimeNet, "WorkerDiskSpace", str(disk))
	# options.archive_dir = archive_dir
	config.set(SEC.PrimeNet, "ProofArchiveDir", archive_dir)
	options.day_night_memory = int(day_night_memory * 1024)
	config.set(SEC.PrimeNet, "Memory", str(options.day_night_memory))
	# # options.cert_cpu_limit = cert_cpu
	# config.set(SEC.PrimeNet, "CertDailyCPULimit", str(cert_cpu))

	num_thread = ask_int("Number of workers (CPU cores or GPUs)", options.num_workers, 1)

	print("""Use the following values to select a work preference:
	2 - Trial factoring
	4 - P-1 factoring
	12 - Trial factoring GPU
	100 - First time LL tests
	101 - Double-check LL tests
	102 - World record LL tests
	104 - 100 million digit LL tests
	106 - Double-check LL tests with zero shift count
	150 - First time PRP tests
	151 - Double-check PRP tests
	152 - World record PRP tests
	153 - 100 million digit PRP tests
	154 - Smallest available first time PRP that needs P-1 factoring
	155 - Double-check using PRP with proof
	156 - Double-check using PRP with proof and nonzero shift count
	160 - First time PRP on Mersenne cofactors
	161 - Double-check PRP on Mersenne cofactors
""")
	print("Not all worktypes are supported by all the GIMPS programs:")
	if sys.stdout.encoding.lower().startswith("utf"):
		print("""
┌──────────┬────────┬────────┬───────┬───────────┬─────────┬───────────────┐
│ Worktype │ Mlucas │ GpuOwl │ PRPLL │ CUDALucas │ CUDAPm1 │ mfaktc/mfakto │
├──────────┴────────┴────────┴───────┴───────────┴─────────┴───────────────┤
│ 4          ✔        ✔                            ✔                       │
│ 12                                                         ✔             │
│ 100        ✔        ✔*       ✔       ✔                                   │
│ 101        ✔                         ✔                                   │
│ 102        ✔        ✔*       ✔       ✔                                   │
│ 104        ✔        ✔*       ✔       ✔                                   │
│ 106                 ✔*       ✔                                           │
│ 150        ✔        ✔        ✔                                           │
│ 151        ✔        ✔        ✔                                           │
│ 152        ✔        ✔        ✔                                           │
│ 153        ✔        ✔        ✔                                           │
│ 154        ✔        ✔*                                                   │
│ 155                 ✔        ✔                                           │
│ 156                                                                      │
│ 160        ✔                                                             │
│ 161        ✔                                                             │
└──────────────────────────────────────────────────────────────────────────┘""")
	else:
		print("""
+----------+--------+--------+-------+-----------+---------+---------------+
| Worktype | Mlucas | GpuOwl | PRPLL | CUDALucas | CUDAPm1 | mfaktc/mfakto |
+----------+--------+--------+-------+-----------+---------+---------------+
| 4          X        X                            X                       |
| 12                                                         X             |
| 100        X        X*       X       X                                   |
| 101        X                         X                                   |
| 102        X        X*       X       X                                   |
| 104        X        X*       X       X                                   |
| 106                 X*       X                                           |
| 150        X        X        X                                           |
| 151        X        X        X                                           |
| 152        X        X        X                                           |
| 153        X        X        X                                           |
| 154        X        X*                                                   |
| 155                 X        X                                           |
| 156                                                                      |
| 160        X                                                             |
| 161        X                                                             |
+--------------------------------------------------------------------------+""")
	print("* Some previous versions of GpuOwl\n")

	work_pref = []

	for i in range(num_thread):
		if num_thread > 1:
			print("\nOptions for worker #{0}\n".format(i + 1))

		work_pref.append(
			ask_str(
				"Type of work to get",
				options.work_preference[i]
				if hasattr(opts_no_defaults, "work_preference") and i < len(options.work_preference)
				else str(PRIMENET.WP_GPU_FACTOR)
				if options.mfaktc or options.mfakto
				else str(PRIMENET.WP_LL_DBLCHK)
				if options.cudalucas
				else str(PRIMENET.WP_PRP_FIRST),
			)
		)

	# cert_work = ask_yn("Get occasional PRP proof certification work", False if options.cert_work is None else options.cert_work)

	if not ask_ok_cancel():
		return None
	if options.num_workers != num_thread:
		options.num_workers = num_thread
		config.set(SEC.PrimeNet, "NumWorkers", str(num_thread))

	options.work_preference = work_pref

	# if cert_work:
	# 	# options.cert_work = cert_work
	# 	config.set(SEC.PrimeNet, "CertWork", str(cert_work))

	work = ask_float(
		"Days of work to queue up",
		(1.0 if options.mfaktc or options.mfakto else 3.0) if options.days_of_work is None else options.days_of_work,
		0,
		90,
	)
	end_dates = ask_int(
		"Hours to wait between sending assignment progress and expected completion dates", options.hours_between_checkins, 1, 7 * 24
	)
	noise = not (config.getboolean(SEC.PrimeNet, "SilentVictory") if config.has_option(SEC.PrimeNet, "SilentVictory") else False)
	noise = ask_yn("Make noise if a new Mersenne prime is found", noise)
	report_100m = ask_yn(
		"Report prime results for exponents greater than or equal to 100 million digits", not options.no_report_100m
	)

	if not ask_ok_cancel():
		return None
	options.days_of_work = work
	config.set(SEC.PrimeNet, "DaysOfWork", str(work))
	# options.hours_between_checkins = end_dates
	config.set(SEC.PrimeNet, "HoursBetweenCheckins", str(end_dates))
	config.set(SEC.PrimeNet, "SilentVictory", str(not noise))
	if not report_100m:
		# options.no_report_100m = not report_100m
		config.set(SEC.PrimeNet, "no_report_100m", str(not report_100m))
	else:
		# options.no_report_100m = None
		config.remove_option(SEC.PrimeNet, "no_report_100m")

	test_email = False
	if ask_yn(
		"Do you want to set the optional e-mail/text message notification settings? (requires providing an SMTP server)",
		options.fromemail and options.smtp,
	):
		smtp_server = ask_str("SMTP server (hostname and optional port), e.g., 'mail.example.com:465'", options.smtp or "")
		tls = ask_yn("Use a secure connection with SSL/TLS?", True if options.tls is None else options.tls)
		starttls = None
		if not tls:
			starttls = ask_yn(
				"Upgrade to a secure connection with StartTLS?", True if options.starttls is None else options.starttls
			)
		fromemail = ask_str("From e-mail address, e.g., 'User <user@example.com>'", options.fromemail or "")
		username = ask_str("Optional username for this account, e.g., 'user@example.com'", options.email_username or "")
		password = ask_pass("Optional password for this account", options.email_password or "")
		toemails = []
		for i in count():
			toemail = (
				ask_str("To e-mail address #{0:n}, e.g., 'User <user@example.com>'".format(i + 1), options.toemails[i])
				if i < len(options.toemails)
				else ask_str(
					"To e-mail address #{0:n}, e.g., 'User <user@example.com>' (leave blank {1})".format(
						i + 1, "to use the From e-mail address" if not i else "to continue"
					),
					"",
				)
			)
			if not toemail:
				break
			toemails.append(toemail)
		test_email = ask_yn("Send a test e-mail message?", True)

		if not ask_ok_cancel():
			return None
		options.smtp = smtp_server
		config.set(SEC.Email, "smtp", smtp_server)
		if tls:
			options.tls = tls
			config.set(SEC.Email, "tls", str(tls))
		if starttls:
			options.starttls = starttls
			config.set(SEC.Email, "starttls", str(starttls))
		options.fromemail = fromemail
		config.set(SEC.Email, "fromemail", fromemail)
		options.email_username = username
		config.set(SEC.Email, "username", username)
		options.email_password = password
		config.set(SEC.Email, "password", password)
		options.toemails = toemails
		config.set(SEC.Email, "toemails", ",".join(toemails))
	return test_email


def readonly_list_file(filename, mode="r"):
	"""Yields lines from a file as strings."""
	# Used when there is no intention to write the file back, so don't
	# check or write lockfiles. Also returns a single string, no list.
	try:
		with io.open(filename, mode, encoding="utf-8") as file:
			for line in file:
				yield line.rstrip("\n")
	except (IOError, OSError):
		# logging.debug("Error reading %r file: %s", filename, e)
		pass


attr_to_copy = {
	SEC.PrimeNet: {
		"worktodo_file": "workfile",
		"results_file": "resultsfile",
		"logfile": "logfile",
		"archive_dir": "ProofArchiveDir",
		"user_id": "username",
		"password": "password",
		"cert_work": "CertWork",
		"cert_cpu_limit": "CertDailyCPULimit",
		"min_exp": "GetMinExponent",
		"max_exp": "GetMaxExponent",
		"min_bit": "bit_min",
		"max_bit": "bit_max",
		"mlucas": "mlucas",
		"gpuowl": "gpuowl",
		"cudalucas": "cudalucas",
		"mfaktc": "mfaktc",
		"mfakto": "mfakto",
		"num_workers": "NumWorkers",
		"num_cache": "num_cache",
		"days_of_work": "DaysOfWork",
		"tests_saved": "tests_saved",
		"pm1_multiplier": "pm1_multiplier",
		"pm1_bounds": "pm1_bounds",
		"no_report_100m": "no_report_100m",
		"convert_ll_to_prp": "convert_ll_to_prp",
		"convert_prp_to_ll": "convert_prp_to_ll",
		"hours_between_checkins": "HoursBetweenCheckins",
		"color": "color",
		"computer_id": "ComputerID",
		"cpu_brand": "CpuBrand",
		"cpu_features": "cpu_features",
		"cpu_speed": "CpuSpeed",
		"memory": "memory",
		"day_night_memory": "Memory",
		"worker_disk_space": "WorkerDiskSpace",
		"cpu_l1_cache_size": "L1",
		"cpu_l2_cache_size": "L2",
		"cpu_l3_cache_size": "L3",
		"num_cores": "NumCores",
		"cpu_hyperthreads": "CpuNumHyperthreads",
		"cpu_hours": "CPUHours",
	},
	SEC.Email: {
		"toemails": "toemails",
		"fromemail": "fromemail",
		"smtp": "smtp",
		"tls": "tls",
		"starttls": "starttls",
		"email_username": "username",
		"email_password": "password",
	},
}

# allows us to give hints for config types that don't have a default optparse value (due to having dynamic defaults)
OPTIONS_TYPE_HINTS = {
	SEC.PrimeNet: {
		"GetMinExponent": int,
		"GetMaxExponent": int,
		"mlucas": bool,
		"gpuowl": bool,
		# "cudalucas": bool,
		"mfaktc": bool,
		"mfakto": bool,
		"CertWork": bool,
		"DaysOfWork": float,
		"tests_saved": float,
		"pm1_multiplier": float,
		"color": bool,
	},
	SEC.Email: {"tls": bool, "starttls": bool},
}


def config_read():
	"""Reads and returns the configuration from the local file, ensuring required sections exist."""
	config = ConfigParser()
	config.optionxform = lambda option: option
	localfile = os.path.join(workdir, options.localfile)
	try:
		config.read([localfile])
	except ConfigParserError as e:
		logging.exception("Error reading %r file: %s", localfile, e, exc_info=options.debug)
	for section in (SEC.PrimeNet, SEC.Email, SEC.Internals):
		if not config.has_section(section):
			# Create the section to avoid having to test for it later
			config.add_section(section)
	return config


def config_write(config, guid=None):
	"""Writes the configuration to a prime.ini file, optionally updating the ComputerGUID."""
	# generate a new prime.ini file
	if guid is not None:  # update the guid if necessary
		config.set(SEC.PrimeNet, "ComputerGUID", guid)
	localfile = os.path.join(workdir, options.localfile)
	with open(localfile, "w") as configfile:
		config.write(configfile)


def get_guid(config):
	"""Retrieve the ComputerGUID from the configuration if it exists."""
	if config.has_option(SEC.PrimeNet, "ComputerGUID"):
		return config.get(SEC.PrimeNet, "ComputerGUID")
	return None


def create_new_guid():
	"""Generate a new GUID (Globally Unique Identifier) as a hexadecimal string."""
	return uuid.uuid4().hex


def merge_config_and_options(config, options):
	"""Synchronizes options with config, updating config if necessary."""
	# getattr and setattr allow access to the options.xxxx values by name
	# which allow to copy all of them programmatically instead of having
	# one line per attribute. Only the attr_to_copy list need to be updated
	# when adding an option you want to copy from argument options to
	# prime.ini config.
	updated = False
	for section, value in attr_to_copy.items():
		for attr, option in value.items():
			# if "attr" has its default value in options, copy it from config
			attr_val = getattr(options, attr)
			type_hint = OPTIONS_TYPE_HINTS[section].get(option)
			if not hasattr(opts_no_defaults, attr) and config.has_option(section, option):
				# If no option is given and the option exists in prime.ini, take it
				# from prime.ini
				if isinstance(attr_val, (list, tuple)) or type_hint in {list, tuple}:
					val = config.get(section, option)
					new_val = val.split(",") if val else []
				elif isinstance(attr_val, bool) or type_hint is bool:
					new_val = config.getboolean(section, option)
				else:
					new_val = config.get(section, option)
				# config file values are always str()
				# they need to be converted to the expected type from options
				if attr_val is not None:
					new_val = type(attr_val)(new_val)
				elif type_hint is not None:
					new_val = type_hint(new_val)
				setattr(options, attr, new_val)
			elif attr_val is not None:
				# If an option is given (even default value) and it is not already
				# identical in prime.ini, update prime.ini
				if isinstance(attr_val, (list, tuple)):
					new_val = ",".join(map(str, attr_val))
				else:
					new_val = str(attr_val)
				if not config.has_option(section, option) or config.get(section, option) != new_val:
					logging.debug(
						"update %r with %s=%s", options.localfile, option, "*" * len(new_val) if "password" in option else new_val
					)
					config.set(section, option, new_val)
					updated = True

	return updated


def is_known_mersenne_prime(p):
	"""Check if a given number is a known Mersenne prime exponent."""
	mersenne_primes = frozenset((
		2,
		3,
		5,
		7,
		13,
		17,
		19,
		31,
		61,
		89,
		107,
		127,
		521,
		607,
		1279,
		2203,
		2281,
		3217,
		4253,
		4423,
		9689,
		9941,
		11213,
		19937,
		21701,
		23209,
		44497,
		86243,
		110503,
		132049,
		216091,
		756839,
		859433,
		1257787,
		1398269,
		2976221,
		3021377,
		6972593,
		13466917,
		20996011,
		24036583,
		25964951,
		30402457,
		32582657,
		37156667,
		42643801,
		43112609,
		57885161,
		74207281,
		77232917,
		82589933,
		136279841,
	))
	return p in mersenne_primes


# https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Testing_against_small_sets_of_bases
# https://oeis.org/A006945
PRIME_BASES = (
	(1, 2047),
	(2, 1373653),
	(3, 25326001),
	(4, 3215031751),
	(5, 2152302898747),
	(6, 3474749660383),
	(7, 341550071728321),
	(9, 3825123056546413051),
	(12, 318665857834031151167461),
	(13, 3317044064679887385961981),
	# Propositions only
	# https://www.ams.org/journals/mcom/2007-76-260/S0025-5718-07-01977-1/S0025-5718-07-01977-1.pdf
	# (14, 6003094289670105800312596501),
	# (15, 59276361075595573263446330101),
	# (16, 564132928021909221014087501701),
	# (18, 1543267864443420616877677640751301),
)


def primes(limit):
	"""Generate a list of prime numbers up to a given limit."""
	if not limit & 1:
		limit -= 1
	size = (limit - 1) // 2
	sieve = bytearray((1,)) * size
	for i in range(isqrt(size) + 1):
		if sieve[i]:
			p = 3 + 2 * i
			j = (p * p - 3) // 2
			# sieve[j : size : p] = bytes(len(range(j, size, p)))
			sieve[j:size:p] = bytearray(len(range(j, size, p)))

	return array("H", chain((2,), (3 + 2 * i for i in range(size) if sieve[i])))


PRIMES = primes(3671)
BASES = PRIMES[: PRIME_BASES[-1][0]]


def miller_rabin(n, nm1, a, d, s):
	"""Performs the Miller-Rabin primality test for a given base 'a'."""
	x = pow(a, d, n)

	if x in {1, nm1}:
		return False

	for _ in range(1, s):
		x = pow(x, 2, n)

		if x == nm1:
			return False
		if x == 1:
			return True

	return True


def is_prime(n):
	"""Check if a number is prime using trial division and the Miller-Rabin primality test."""
	if n < 2:
		return False
	for p in BASES:
		if n == p:
			return True
		if not n % p:
			return False

	d = nm1 = n - 1
	r = 0
	while not d & 1:
		d >>= 1
		r += 1

	for i, num in PRIME_BASES:
		if n < num:
			bases = BASES[:i]
			break
	else:
		idx = n.bit_length() >> 1
		bases = PRIMES[:idx]

	return not any(miller_rabin(n, nm1, a, d, r) for a in bases)


if sys.version_info >= (3, 3):
	import decimal

	def digits(assignment):
		"""Calculate the number of decimal digits in the given assignment."""
		# Maximum exponent on 32-bit systems: 1,411,819,440 (425,000,000 digits)
		exponent = exponent_to_str(assignment)
		adigits = int(Decimal(assignment.k).log10() + assignment.n * Decimal(assignment.b).log10()) + 1
		if adigits <= 300000000:
			logging.debug("Calculating the number of digits for %s…", exponent)
			with decimal.localcontext() as ctx:
				ctx.prec = decimal.MAX_PREC
				ctx.Emax = decimal.MAX_EMAX
				ctx.Emin = decimal.MIN_EMIN
				ctx.traps[decimal.Inexact] = True

				num = str(int(assignment.k) * Decimal(assignment.b) ** assignment.n + assignment.c)
				adigits = len(num)
				logging.info(
					"The exponent %s has %s decimal digits: %s",
					exponent,
					format(adigits, "n"),
					"{0}…{1}".format(num[:20], num[-20:]) if adigits > 50 else num,
				)
		else:
			logging.info(
				"The exponent %s has approximately %s decimal digits (using formula log10(%s) + %s * log10(%s) + 1)",
				exponent,
				format(adigits, "n"),
				assignment.k,
				assignment.n,
				assignment.b,
			)
		return adigits

else:

	def digits(assignment):
		"""Calculate the number of decimal digits in the given assignment."""
		adigits = int(Decimal(assignment.k).log10() + assignment.n * Decimal(assignment.b).log10()) + 1
		logging.info(
			"The exponent %s has approximately %s decimal digits (using formula log10(%s) + %s * log10(%s) + 1)",
			exponent_to_str(assignment),
			format(adigits, "n"),
			assignment.k,
			assignment.n,
			assignment.b,
		)
		return adigits


WORKPATTERN = re.compile(
	r'^(?:(?:B1=([0-9]+)(?:,B2=([0-9]+))?|B2=([0-9]+));)?(Test|DoubleCheck|PRP(?:DC)?|Factor|P[Ff]actor|P[Mm]inus1|Cert)\s*=\s*(?:(([0-9A-F]{32})|[Nn]/[Aa]|0),)?(?:([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)|"[0-9]+(?:,[0-9]+)*")(?:,|$)){1,9}$'
)

Test_RE = re.compile(
	r"^(?:(?:B1=[0-9]+(?:,B2=[0-9]+)?|B2=[0-9]+);)?(Test|DoubleCheck)\s*=\s*(?:([0-9A-F]{32}|[Nn]/[Aa]|0),)?([0-9]+)(?:,([0-9]+),([0-9]+))?$"
)
PRP_RE = re.compile(
	r'^(?:(?:B1=[0-9]+(?:,B2=[0-9]+)?|B2=[0-9]+);)?(PRP(?:DC)?)\s*=\s*(?:([0-9A-F]{32}|[Nn]/[Aa]|0),)?([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)),([0-9]+),([0-9]+),([-+]?[0-9]+)(?:,([0-9]+),([0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:,([0-9]+),([0-9]+))?)?(?:,"([0-9]+(?:,[0-9]+)*)")?$'
)
Factor_RE = re.compile(r"^(Factor)\s*=\s*(?:([0-9A-F]{32}|[Nn]/[Aa]|0),)?([0-9]+),([0-9]+),([0-9]+)$")
PFactor_RE = re.compile(
	r'^(?:(?:B1=[0-9]+(?:,B2=[0-9]+)?|B2=[0-9]+);)?(P[Ff]actor)\s*=\s*(?:([0-9A-F]{32}|[Nn]/[Aa]|0),)?([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)),([0-9]+),([0-9]+),([-+]?[0-9]+),([0-9]+),([0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:,"([0-9]+(?:,[0-9]+)*)")?$'
)
PMinus1_RE = re.compile(
	r'^(P[Mm]inus1)\s*=\s*(?:([0-9A-F]{32}|[Nn]/[Aa]|0),)?([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)),([0-9]+),([0-9]+),([-+]?[0-9]+),([0-9]+),([0-9]+)(?:,([0-9]+)(?:,([0-9]+))?)?(?:,"([0-9]+(?:,[0-9]+)*)")?$'
)
Cert_RE = re.compile(
	r"^(Cert)\s*=\s*(?:([0-9A-F]{32}|[Nn]/[Aa]|0),)?([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)),([0-9]+),([0-9]+),([-+]?[0-9]+),([0-9]+)$"
)


def parse_assignment(task):
	"""Parses an assignment string and returns an Assignment object with the extracted details."""
	# Ex: Test=197ED240A7A41EC575CB408F32DDA661,57600769,74
	found = WORKPATTERN.match(task)
	if not found:
		return None
	# logging.debug(task)
	assignment = Assignment()
	B1, B21, B22, work_type, value, assignment.uid = found.group(1, 2, 3, 4, 5, 6)
	if B1:
		assignment.B1 = int(B1)
		if B21:
			assignment.B2 = int(B21)
	if B22:
		assignment.B2 = int(B22)
	assignment.ra_failed = bool(value) and not assignment.uid
	# e.g., "57600769", "197ED240A7A41EC575CB408F32DDA661"
	# logging.debug("type = %s, assignment_id = %s", work_type, assignment.uid)
	# Extract the subfield containing the exponent, whose position depends on
	# the assignment type:
	if work_type in {"Test", "DoubleCheck"}:
		found = Test_RE.match(task)
		if not found:
			return None
		_, _, n, sieve_depth, pminus1ed = found.groups()
		assignment.work_type = PRIMENET.WORK_TYPE_FIRST_LL if work_type == "Test" else PRIMENET.WORK_TYPE_DBLCHK
		assignment.n = int(n)
		if pminus1ed:
			assignment.sieve_depth = float(sieve_depth)
			assignment.pminus1ed = int(pminus1ed)
		# assignment.tests_saved = 2.0 if assignment.work_type == PRIMENET.WORK_TYPE_FIRST_LL else 1.0
	elif work_type in {"PRP", "PRPDC"}:
		found = PRP_RE.match(task)
		if not found:
			return None
		_, _, k, b, n, c, sieve_depth, tests_saved, prp_base, prp_residue_type, known_factors = found.groups()
		assignment.prp_dblchk = work_type == "PRPDC"
		assignment.work_type = PRIMENET.WORK_TYPE_PRP
		assignment.k = float(k)
		assignment.b = int(b)
		assignment.n = int(n)
		assignment.c = int(c)
		if tests_saved:
			assignment.sieve_depth = float(sieve_depth)
			assignment.tests_saved = float(tests_saved)
			if prp_residue_type:
				assignment.prp_base = int(prp_base)
				assignment.prp_residue_type = int(prp_residue_type)
		if known_factors:
			assignment.known_factors = tuple(map(int, known_factors.split(",")))
	elif work_type == "Factor":
		found = Factor_RE.match(task)
		if not found:
			return None
		_, _, n, sieve_depth, factor_to = found.groups()
		assignment.work_type = PRIMENET.WORK_TYPE_FACTOR
		assignment.n = int(n)
		assignment.sieve_depth = float(sieve_depth)
		assignment.factor_to = float(factor_to)
	elif work_type in {"PFactor", "Pfactor"}:
		found = PFactor_RE.match(task)
		if not found:
			return None
		_, _, k, b, n, c, sieve_depth, tests_saved, known_factors = found.groups()
		assignment.work_type = PRIMENET.WORK_TYPE_PFACTOR
		assignment.k = float(k)
		assignment.b = int(b)
		assignment.n = int(n)
		assignment.c = int(c)
		assignment.sieve_depth = float(sieve_depth)
		assignment.tests_saved = float(tests_saved)
		if known_factors:
			assignment.known_factors = tuple(map(int, known_factors.split(",")))
	elif work_type in {"PMinus1", "Pminus1"}:
		found = PMinus1_RE.match(task)
		if not found:
			return None
		_, _, k, b, n, c, B1, B2, sieve_depth, B2_start, known_factors = found.groups()
		assignment.work_type = PRIMENET.WORK_TYPE_PMINUS1
		assignment.k = float(k)
		assignment.b = int(b)
		assignment.n = int(n)
		assignment.c = int(c)
		assignment.B1 = int(B1)
		assignment.B2 = int(B2)
		assignment.sieve_depth = 0.0
		if sieve_depth:
			assignment.sieve_depth = float(sieve_depth)
			if B2_start:
				assignment.B2_start = int(B2_start)
		if known_factors:
			assignment.known_factors = tuple(map(int, known_factors.split(",")))
	elif work_type == "Cert":
		found = Cert_RE.match(task)
		if not found:
			return None
		_, _, k, b, n, c, cert_squarings = found.groups()
		assignment.work_type = PRIMENET.WORK_TYPE_CERT
		assignment.k = float(k)
		assignment.b = int(b)
		assignment.n = int(n)
		assignment.c = int(c)
		assignment.cert_squarings = int(cert_squarings)
	if assignment.n and assignment.n >= MAX_PRIMENET_EXP:
		assignment.ra_failed = True
	return assignment


def process_add_file(adapter, adir):
	"""Processes and appends tasks from an .add file to the work file, then removes the .add file."""
	workfile = os.path.join(adir, options.worktodo_file)
	addfile = os.path.splitext(workfile)[0] + ".add"  # ".add.txt"
	if os.path.exists(addfile):
		with LockFile(addfile), io.open(workfile, "a", encoding="utf-8") as file:
			add = readonly_list_file(addfile)
			for task in add:
				adapter.debug("Adding %r line to the %r file", task, workfile)
				file.write(task + "\n")
			os.remove(addfile)


def read_workfile(adapter, adir):
	"""Reads and validates assignments from a work file, yielding the assignments."""
	workfile = os.path.join(adir, options.worktodo_file)
	tasks = readonly_list_file(workfile)
	for task in tasks:
		illegal_line = False
		assignment = parse_assignment(task)
		if assignment is not None:
			if (
				assignment.k == 1.0
				and assignment.b == 2
				and not is_prime(assignment.n)
				and assignment.c == -1
				and not assignment.known_factors
				and assignment.work_type != PRIMENET.WORK_TYPE_PMINUS1
			):
				adapter.error("%r file contained composite exponent: %s.", workfile, assignment.n)
				illegal_line = True
			if assignment.work_type == PRIMENET.WORK_TYPE_PMINUS1 and assignment.B1 < 50000:
				adapter.error("%r file has P-1 with B1 < 50000 (exponent: %s).", workfile, assignment.n)
				illegal_line = True
		else:
			illegal_line = True
		if illegal_line:
			adapter.error("Illegal line in %r file: %r", workfile, task)
			yield task
		else:
			yield assignment


def output_assignment(assignment):
	"""Generate a formatted string representing the details of a given assignment."""
	temp = []
	if assignment.uid:
		temp.append(assignment.uid)
	elif assignment.ra_failed:
		temp.append("N/A")

	if assignment.work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK}:
		test = "Test" if assignment.work_type == PRIMENET.WORK_TYPE_FIRST_LL else "DoubleCheck"
		temp.append(assignment.n)
		if assignment.sieve_depth != 99.0 or assignment.pminus1ed != 1:
			temp += ("{0:.0f}".format(assignment.sieve_depth), assignment.pminus1ed)
	elif assignment.work_type == PRIMENET.WORK_TYPE_PRP:
		test = "PRP" + ("DC" if assignment.prp_dblchk else "")
		temp += ("{0:.0f}".format(assignment.k), assignment.b, assignment.n, assignment.c)
		if assignment.sieve_depth != 99.0 or assignment.tests_saved > 0.0 or assignment.prp_base or assignment.prp_residue_type:
			temp += ("{0:g}".format(assignment.sieve_depth), "{0:g}".format(assignment.tests_saved))
			if assignment.prp_base or assignment.prp_residue_type:
				temp += (assignment.prp_base, assignment.prp_residue_type)
		if assignment.known_factors:
			temp.append('"' + ",".join(map(str, assignment.known_factors)) + '"')
	elif assignment.work_type == PRIMENET.WORK_TYPE_PFACTOR:
		test = "Pfactor"
		temp += (
			"{0:.0f}".format(assignment.k),
			assignment.b,
			assignment.n,
			assignment.c,
			"{0:g}".format(assignment.sieve_depth),
			"{0:g}".format(assignment.tests_saved),
		)
		if assignment.known_factors:
			temp.append('"' + ",".join(map(str, assignment.known_factors)) + '"')
	elif assignment.work_type == PRIMENET.WORK_TYPE_FACTOR:
		test = "Factor"
		temp += (assignment.n, "{0:.0f}".format(assignment.sieve_depth), "{0:.0f}".format(assignment.factor_to))
	elif assignment.work_type == PRIMENET.WORK_TYPE_PMINUS1:
		test = "Pminus1"
		temp += ("{0:.0f}".format(assignment.k), assignment.b, assignment.n, assignment.c, assignment.B1, assignment.B2)
		if assignment.sieve_depth > 0.0:
			temp.append("{0:.0f}".format(assignment.sieve_depth))
		if assignment.B2_start > assignment.B1:
			temp.append(assignment.B2_start)
		if assignment.known_factors:
			temp.append('"' + ",".join(map(str, assignment.known_factors)) + '"')
	elif assignment.work_type == PRIMENET.WORK_TYPE_CERT:
		test = "Cert"
		temp += ("{0:.0f}".format(assignment.k), assignment.b, assignment.n, assignment.c, assignment.cert_squarings)

	if assignment.work_type != PRIMENET.WORK_TYPE_PMINUS1:
		if assignment.B1:
			test = "B1={0}{1};".format(assignment.B1, ",B2={0}".format(assignment.B2) if assignment.B2 else "") + test
		elif assignment.B2:
			test = "B2={0};".format(assignment.B2) + test
	return test + "=" + ",".join(map(str, temp))


def write_workfile(adir, assignments):
	"""Writes assignments to a work file in the specified directory."""
	workfile = os.path.join(adir, options.worktodo_file)
	tasks = (output_assignment(task) if isinstance(task, Assignment) else task for task in assignments)
	with tempfile.NamedTemporaryFile("w", dir=adir, delete=False) as f:  # Python 3+: encoding="utf-8"
		pass
	with io.open(f.name, "w", encoding="utf-8") as file:
		file.writelines(task + "\n" for task in tasks)
	replace(f.name, workfile)


def announce_prime_to_user(exponent, worktype):
	"""Announces a prime or probable prime number to the user and prompts to send an email."""
	color = BOLD + COLORS.RED if COLOR else ""
	reset = RESET_All if COLOR else ""
	emails = ", ".join(starmap("{0} <{1}>".format, CCEMAILS))
	while True:
		if worktype == "LL":
			print("{0}New Mersenne Prime!!!! M{1} is prime!{2}".format(color, exponent, reset))
		else:
			print("{0}New Probable Prime!!!! {1} is a probable prime!{2}".format(color, exponent, reset))
		print("Please send e-mail to {0}.".format(emails))
		beep()
		time.sleep(1)


def tail(filename, lines=100):
	"""Returns the last 'lines' lines from a file, or an appropriate message if the file is not found or empty."""
	if not os.path.exists(filename):
		return "> (File not found)"
	w = deque(readonly_list_file(filename), lines)
	if not w:
		return "> (File is empty)"
	return "\n".join("> " + line for line in w)


def send(subject, message, attachments=None, to=None, cc=None, bcc=None, priority=None):
	"""Send an email with optional attachments and specified recipients."""
	msg_text = MIMEText(message, "plain", "utf-8")

	if attachments:
		msg = MIMEMultipart()
		msg.attach(msg_text)

		for filename, file in attachments:
			ctype, encoding = mimetypes.guess_type(filename)  # guess_file_type(filename)
			if ctype is None or encoding is not None:
				ctype = "application/octet-stream"
			maintype, subtype = ctype.split("/", 1)
			msg_attach = MIMEBase(maintype, subtype)
			msg_attach.set_payload(file)
			encoders.encode_base64(msg_attach)
			msg_attach.add_header("Content-Disposition", "attachment", filename=("utf-8", "", os.path.basename(filename)))
			msg.attach(msg_attach)
	else:
		msg = msg_text

	COMMASPACE = ", "
	msg["User-Agent"] = "AutoPrimeNet assignment handler version {0}".format(VERSION)
	name, from_addr = fromemail
	msg["From"] = formataddr((Header(name, "utf-8").encode(), from_addr))
	to = toemails + to if to else toemails
	msg["To"] = (
		"undisclosed-recipients:;"
		if not to and not cc
		else COMMASPACE.join(formataddr((Header(name, "utf-8").encode(), addr)) for name, addr in to)
	)
	if cc:
		msg["Cc"] = COMMASPACE.join(formataddr((Header(name, "utf-8").encode(), addr)) for name, addr in cc)
	msg["Subject"] = Header(subject, "utf-8")
	msg["Date"] = formatdate(localtime=True)
	if priority:
		msg["X-Priority"] = priority
	to_addrs = [addr for f in (to, cc, bcc) if f for _, addr in f]

	# Debug code
	# print(msg.as_string())
	# print(msg)

	s = None
	try:
		if options.tls:
			# Python 3.3+
			# with smtplib.SMTP_SSL(options.smtp, context=context, timeout=30) as s:
			s = smtplib.SMTP_SSL(options.smtp, timeout=30)
			if options.debug > 1:
				s.set_debuglevel(2)
			if options.email_username:
				s.login(options.email_username, options.email_password)
			s.sendmail(from_addr, to_addrs, msg.as_string())
		else:
			# Python 3.3+
			# with smtplib.SMTP(options.smtp, timeout=30) as s:
			s = smtplib.SMTP(options.smtp, timeout=30)
			if options.debug > 1:
				s.set_debuglevel(2)
			if options.starttls:
				# Python 3.3+
				# s.starttls(context=context)
				s.starttls()
			if options.email_username:
				s.login(options.email_username, options.email_password)
			s.sendmail(from_addr, to_addrs, msg.as_string())
	except (smtplib.SMTPException, IOError, OSError) as e:  # socket.error
		logging.exception("Failed to send e-mail: %s", e, exc_info=options.debug)
		return False
	finally:
		if s is not None:
			s.quit()
	return True


def send_msg(subject, message="", attachments=None, to=None, cc=None, bcc=None, priority=None, azipfile=None):
	"""Send an email with the specified subject, message, and attachments."""
	if not options.fromemail or not options.smtp:
		return False
	if config.has_option(SEC.Email, "send") and not config.getboolean(SEC.Email, "send"):
		return False
	logging.info("Sending e-mail: %s", subject)

	if attachments:
		aattachments = []
		for attachment in attachments:
			if not isinstance(attachment, (tuple, list)) and not os.path.exists(attachment):
				logging.info("Skipping attachment: cannot read %r file.", attachment)
				message += "(Unable to attach the {0!r} file, as it does not exist.)\n".format(attachment)
			else:
				aattachments.append(attachment)
		attachments = aattachments

		if azipfile:
			if os.path.exists(azipfile):
				logging.info("File %r already exists.", azipfile)
			with zipfile.ZipFile(azipfile, "w") as myzip:
				for attachment in attachments:
					if isinstance(attachment, (tuple, list)):
						filename, file = attachment
						myzip.writestr(os.path.basename(filename), file)
					else:
						myzip.write(attachment, os.path.basename(attachment))
			attachments = [azipfile]

		for i, attachment in enumerate(attachments):
			if not isinstance(attachment, (tuple, list)):
				with open(attachment, "rb") as f:
					attachments[i] = (attachment, f.read())

		logging.debug("Attachments:")
		total = 0
		aattachments = []
		for attachment in attachments:
			filename, file = attachment
			size = len(file)
			total += size
			logging.debug("%r: %s", filename, outputunit(size))
			aattachments.append((size, attachment))

		logging.debug("Total Size: %s", outputunit(total))

		if total >= 25 * 1024 * 1024:
			logging.warning("The total size of all attachments is greater than 25 MiB.")
			total = 0
			for size, attachment in sorted(aattachments):
				filename, _ = attachment
				total += size
				if total >= 25 * 1024 * 1024:
					logging.info("Skipping attachment: %r", filename)
					message += "(Unable to attach the {0!r} file ({1}), as the total message size would be too large.)\n".format(
						filename, outputunit(size)
					)
					attachments.remove(attachment)

	return send(subject, message, attachments, to, cc, bcc, priority)


def test_msg(guid):
	"""Sends a test email to verify AutoPrimeNet email configuration."""
	if not send_msg(
		"👋 Test from AutoPrimeNet",
		"""Hello {0},

This is the requested test message from AutoPrimeNet! You have successfully configured the program to send e-mail notifications.

Version: {1}
Requests/urllib3 library version: {2}/{3}
Python version: {4}

PrimeNet User ID: {5}
Computer name: {6}
GUID: {7}
""".format(
			config.get(SEC.PrimeNet, "user_name") if config.has_option(SEC.PrimeNet, "user_name") else options.user_id,
			VERSION,
			requests.__version__,
			urllib3.__version__,
			platform.python_version(),
			options.user_id,
			options.computer_id,
			guid,
		),
	):
		logging.error(
			"Failed to send test e-mail. Check your configuration or try providing the --debug option twice for more information."
		)
		return False
	return True


def generate_application_str():
	"""Generates a formatted application string based on the platform and selected program."""
	if sys.platform == "darwin":
		aplatform = "Mac OS X" + (" 64-bit" if is_64bit else "")
	else:
		aplatform = platform.system() + ("64" if is_64bit else "")
	program = PROGRAMS[
		0
		if options.prime95
		else 5
		if options.mfakto
		else 4
		if options.mfaktc
		else 3
		if options.cudalucas
		else 2
		if options.gpuowl
		else 1
	]
	if options.prime95:
		return "{0},{1[name]},v{1[version]},build {1[build]}".format(aplatform, program)
	name = program["name"]
	version = program["version"]
	if config.has_option(SEC.Internals, "program"):
		aprogram = config.get(SEC.Internals, "program").split(None, 1)
		if len(aprogram) == 2:
			name, version = aprogram
			# Python 3.9+: version = version.removeprefix("v")
			version = version[1:] if version.startswith("v") else version
	# return "{0},{1},v{2};Python {3},{4}".format(
	#     aplatform, name, version, platform.python_version(), parser.get_version())
	return "{0},{1},v{2}".format(aplatform, name, version)


def get_os():
	"""Retrieve detailed information about the operating system."""
	result = {}
	machine = None

	if sys.platform == "win32":
		result["os"] = "Windows"
		release, version, csd, _ptype = platform.win32_ver()
		if release:
			result["release"] = release
			result["build"] = version
			if csd != "SP0":
				result["service-pack"] = csd
	elif sys.platform == "darwin":
		result["os"] = "macOS"
		release, _versioninfo, machine = platform.mac_ver()
		if release:
			result["release"] = release
	elif sys.platform.startswith("linux"):
		result["os"] = "Linux"
		try:
			info = freedesktop_os_release()
		except OSError:
			pass
		else:
			if info:
				result["name"] = info["NAME"]
				if "VERSION" in info:
					result["version"] = info["VERSION"]
				else:
					result["distribution"] = info["PRETTY_NAME"]
		# Python 2.6 - 3.7
		if "name" not in result and hasattr(platform, "linux_distribution"):
			distname, version, _id = platform.linux_distribution()
			if distname:
				result["name"] = distname
				result["version"] = version

	if not machine:
		machine = platform.machine()
	if machine:
		result["architecture"] = machine

	return result


def get_cpu_model():
	"""Returns the model name of the CPU."""
	output = ""
	if sys.platform == "win32":
		# wmic cpu get name
		# (Get-CimInstance Win32_Processor).Name
		try:
			with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
				output, _ = winreg.QueryValueEx(key, "ProcessorNameString")
		except WindowsError:
			pass
	elif sys.platform == "darwin":
		output = sysctl_str(b"machdep.cpu.brand_string").decode("utf-8")
	elif sys.platform.startswith("linux"):
		with open("/proc/cpuinfo") as f:
			for line in f:
				if line.startswith("model name"):
					output = re.sub(r"^.*: *", "", line.rstrip(), 1)
					break
	return output


def get_cpu_cores_threads():
	"""Returns the number of physical CPU cores and logical threads available on the system."""
	# Python 3.4+, but can be overridden in 3.13+
	# threads = os.cpu_count()
	# threads = multiprocessing.cpu_count()
	cores = threads = 0
	if sys.platform == "win32":
		# wmic cpu get NumberOfCores,NumberOfLogicalProcessors
		# Get-CimInstance Win32_Processor | Select NumberOfCores,NumberOfLogicalProcessors
		return_length = wintypes.DWORD()
		# Windows 7 or greater
		if hasattr(kernel32, "GetLogicalProcessorInformationEx"):
			RelationAll = 0xFFFF
			if (
				not kernel32.GetLogicalProcessorInformationEx(RelationAll, None, ctypes.byref(return_length))
				and ctypes.get_last_error() == 122  # ERROR_INSUFFICIENT_BUFFER
			):
				buffer = ctypes.create_string_buffer(return_length.value)
				if kernel32.GetLogicalProcessorInformationEx(RelationAll, buffer, ctypes.byref(return_length)):
					offset = 0
					while offset < return_length.value:
						ptr = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX.from_buffer(buffer, offset)
						if ptr.Relationship == 0:  # RelationProcessorCore
							cores += 1
						offset += ptr.Size
		# ERROR_INSUFFICIENT_BUFFER
		elif not kernel32.GetLogicalProcessorInformation(None, ctypes.byref(return_length)) and ctypes.get_last_error() == 122:
			buffer = (
				SYSTEM_LOGICAL_PROCESSOR_INFORMATION * (return_length.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION))
			)()
			if kernel32.GetLogicalProcessorInformation(buffer, ctypes.byref(return_length)):
				cores = sum(1 for ptr in buffer if ptr.Relationship == 0)  # RelationProcessorCore
		# raise ctypes.WinError()
		threads = int(os.getenv("NUMBER_OF_PROCESSORS", "0"))
	elif sys.platform == "darwin":
		cores = sysctl_value(b"hw.physicalcpu_max", ctypes.c_int)
		threads = sysctl_value(b"hw.logicalcpu_max", ctypes.c_int)
	elif sys.platform.startswith("linux"):
		acores = set()
		for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_cpus_list") or glob.glob(
			"/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list"
		):
			with open(path) as f:
				acores.add(f.read().rstrip())
		cores = len(acores)
		threads = os.sysconf("SC_NPROCESSORS_CONF" if sys.version_info >= (3,) else b"SC_NPROCESSORS_CONF")
	return cores, threads


def get_cpu_frequency():
	"""Retrieve the maximum CPU frequency in MHz for the system."""
	frequency = 0
	if sys.platform == "win32":
		# wmic cpu get MaxClockSpeed
		# (Get-CimInstance Win32_Processor).MaxClockSpeed
		try:
			with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
				frequency, _ = winreg.QueryValueEx(key, "~MHz")
		except WindowsError:
			pass
	elif sys.platform == "darwin":
		output = sysctl_value(b"hw.cpufrequency_max", ctypes.c_uint64)
		if output:
			frequency = output // 1000 // 1000
	elif sys.platform.startswith("linux"):
		with open("/proc/cpuinfo") as f:
			freqs = [float(re.sub(r"^.*: *", "", line.rstrip(), 1)) for line in f if line.startswith("cpu MHz")]
		if freqs:
			freq = set(freqs)
			if len(freq) == 1:
				frequency = int(freq.pop())
	return frequency


def get_physical_memory():
	"""Returns the total physical memory in MiB of the system."""
	memory = 0
	if sys.platform == "win32":
		# wmic memphysical get MaxCapacity
		# wmic ComputerSystem get TotalPhysicalMemory
		# (Get-CimInstance Win32_PhysicalMemoryArray).MaxCapacity
		# (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
		memory_status = MEMORYSTATUSEX()
		ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
		memory = memory_status.ullTotalPhys >> 20
	elif sys.platform == "darwin":
		output = sysctl_value(b"hw.memsize", ctypes.c_uint64)
		if output:
			memory = output >> 20
	elif sys.platform.startswith("linux"):
		# os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
		info = sysinfo()
		if not libc.sysinfo(ctypes.byref(info)):
			memory = (info.totalram * info.mem_unit) >> 20
	return memory


def get_cpu_cache_sizes():
	"""Retrieve the sizes of the CPU caches (L1, L2, L3) for the system."""
	cache_sizes = {1: 0, 2: 0, 3: 0}
	if sys.platform == "win32":
		# wmic cpu get L2CacheSize,L3CacheSize
		# wmic path Win32_CacheMemory get CacheType,InstalledSize,Level
		# Get-CimInstance Win32_Processor | Select L2CacheSize,L3CacheSize
		# Get-CimInstance Win32_CacheMemory | Select CacheType,InstalledSize,Level
		return_length = wintypes.DWORD()
		# Windows 7 or greater
		if hasattr(kernel32, "GetLogicalProcessorInformationEx"):
			RelationAll = 0xFFFF
			if (
				not kernel32.GetLogicalProcessorInformationEx(RelationAll, None, ctypes.byref(return_length))
				and ctypes.get_last_error() == 122  # ERROR_INSUFFICIENT_BUFFER
			):
				buffer = ctypes.create_string_buffer(return_length.value)
				if kernel32.GetLogicalProcessorInformationEx(RelationAll, buffer, ctypes.byref(return_length)):
					offset = 0
					while offset < return_length.value:
						ptr = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX.from_buffer(buffer, offset)
						offset += ptr.Size
						if ptr.Relationship == 2:  # RelationCache
							if ptr.Cache.Type == 1:  # CacheInstruction
								continue
							cache_sizes[ptr.Cache.Level] = ptr.Cache.CacheSize
		# ERROR_INSUFFICIENT_BUFFER
		elif not kernel32.GetLogicalProcessorInformation(None, ctypes.byref(return_length)) and ctypes.get_last_error() == 122:
			buffer = (
				SYSTEM_LOGICAL_PROCESSOR_INFORMATION * (return_length.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION))
			)()
			if kernel32.GetLogicalProcessorInformation(buffer, ctypes.byref(return_length)):
				for ptr in buffer:
					if ptr.Relationship == 2:  # RelationCache
						if ptr.Cache.Type == 1:  # CacheInstruction
							continue
						cache_sizes[ptr.Cache.Level] = ptr.Cache.Size
		# raise ctypes.WinError()
	elif sys.platform == "darwin":
		for level, cache in enumerate((b"hw.l1dcachesize", b"hw.l2cachesize", b"hw.l3cachesize"), 1):
			output = sysctl_value(cache, ctypes.c_int)
			if output:
				cache_sizes[level] = output
	elif sys.platform.startswith("linux"):
		for path in glob.iglob("/sys/devices/system/cpu/cpu[0-9]*/cache"):
			for file in glob.iglob(os.path.join(path, "index[0-9]*/size")):
				with open(file) as f:
					size = inputunit(f.read().rstrip())
				adir = os.path.dirname(file)
				with open(os.path.join(adir, "level")) as f:
					level = int(f.read().rstrip())
				with open(os.path.join(adir, "type")) as f:
					atype = f.read().rstrip()
				if atype == "Instruction":
					continue
				cache_sizes[level] = size
	for cache in cache_sizes:
		cache_sizes[cache] >>= 10
	return cache_sizes


def parse_v5_resp(r):
	"""Parses a v5 response string into a dictionary of options and values."""
	ans = {}
	for line in r.split("\n"):
		if line == "==END==":
			break
		option, _, value = line.partition("=")
		ans[option] = value.replace("\r", "\n")
	return ans


__v5salt_ = 0


def secure_v5_url(guid, args):
	"""Generates a secure v5 URL with a hash based on the provided GUID and arguments."""
	k = bytearray(md5(guid.encode("utf-8")).digest())

	for i in range(16):
		k[i] ^= k[(k[i] ^ _V5_UNIQUE_TRUSTED_CLIENT_CONSTANT_ & 0xFF) % 16] ^ _V5_UNIQUE_TRUSTED_CLIENT_CONSTANT_ // 256

	p_v5key = md5(k).hexdigest().upper()

	global __v5salt_
	if not __v5salt_:
		random.seed()

	__v5salt_ = random.randint(0, sys.maxsize) & 0xFFFF

	args["ss"] = __v5salt_
	URL = urlencode(args) + "&" + p_v5key

	ahash = md5(URL.encode("utf-8")).hexdigest().upper()

	args["sh"] = ahash


def send_request(guid, args):
	"""Send a request to the PrimeNet server and handle the response."""
	if guid is not None:
		if not options.prime95:
			args["ss"] = 19191919
			args["sh"] = "ABCDABCDABCDABCDABCDABCDABCDABCD"
		else:
			secure_v5_url(guid, args)
	# logging.debug("Args: %s", args)

	try:
		r = session.get(primenet_v5_burl, params=args, timeout=180)
		# logging.debug("URL: " + r.url)
		r.raise_for_status()
		text = r.text
	except Timeout as e:
		logging.exception("%s", e, exc_info=options.debug)
		return None
	except HTTPError as e:
		logging.exception("Error receiving answer to request: %s", e, exc_info=options.debug)
		return None
	except ConnectionError as e:
		logging.exception("Error connecting to server for request: %s", e, exc_info=options.debug)
		return None

	result = parse_v5_resp(text)
	# logging.debug("RESPONSE:\n" + text)
	if "pnErrorResult" not in result:
		logging.error("PnErrorResult value missing.  Full response was:\n%s", text)
		return None
	if "pnErrorDetail" not in result:
		logging.error("PnErrorDetail string missing")
		return None
	rc = int(result["pnErrorResult"])
	if rc:
		resmsg = ERRORS.get(rc, "Unknown error code")
		if args["t"] == "ga" and args.get("cert"):
			logging.debug("PrimeNet error %s: %s", rc, resmsg)
			logging.debug(result["pnErrorDetail"])
		else:
			logging.error("PrimeNet error %s: %s", rc, resmsg)
			logging.error(result["pnErrorDetail"])
	elif result["pnErrorDetail"] != "SUCCESS":
		if result["pnErrorDetail"].count("\n"):
			logging.info("PrimeNet success code with additional info:")
			logging.info(result["pnErrorDetail"])
		else:
			logging.info("PrimeNet success code with additional info: %s", result["pnErrorDetail"])

	return result


def get_exponent(n):
	"""Fetches and returns the JSON data for a given Mersenne exponent."""
	try:
		# r = session.get(primenet_baseurl + "report_exponent_simple/", params={"exp_lo": n, "faclim": 1, "json": 1}, timeout=180)
		r = session.get(mersenne_ca_baseurl + "exponent/{0}/json".format(n), timeout=180)
		r.raise_for_status()
		result = r.json()
	except RequestException as e:
		logging.exception("%s", e, exc_info=options.debug)
		return None

	return result


FACTOR_LIMITS = (
	(82, 1071000000),
	(81, 842000000),
	(80, 662000000),
	(79, 516800000),
	(78, 408400000),
	(77, 322100000),
	(76, 253500000),
	(75, 199500000),
	(74, 153400000),
	(73, 120000000),
	(72, 96830000),
	(71, 77910000),
	(70, 60940000),
	(69, 48800000),
	(68, 38300000),
	(67, 29690000),
	(66, 23390000),
	(65, 13380000),
	(64, 8250000),
	(63, 6515000),
	(62, 5160000),
	(61, 3960000),
	(60, 2950000),
	(59, 2360000),
	(58, 1930000),
	(57, 1480000),
	(56, 1000000),
)


def factor_limit(p):
	"""Determine the factor limit based on the given exponent."""
	test = 40
	for bits, exponent in FACTOR_LIMITS:
		if p > exponent:
			test = bits
			break
	return test + 4


# Adapted from Mihai Preda's script: https://github.com/preda/gpuowl/blob/d8bfa25366bef4178dbd2059e2ba2a3bf3b6e0f0/pm1/pm1.py

# Table of values of Dickman's "rho" function for argument from 2 in steps of 1/20.
rhotab = (
	0.306852819440055,
	0.282765004395792,
	0.260405780162154,
	0.239642788276221,
	0.220357137908328,
	0.202441664262192,
	0.185799461593866,
	0.170342639724018,
	0.155991263872504,
	0.142672445952511,
	0.130319561832251,
	0.118871574006370,
	0.108272442976271,
	0.0984706136794386,
	0.0894185657243129,
	0.0810724181216677,
	0.0733915807625995,
	0.0663384461579859,
	0.0598781159863707,
	0.0539781578442059,
	0.0486083882911316,
	0.0437373330511146,
	0.0393229695406371,
	0.0353240987411619,
	0.0317034445117801,
	0.0284272153221808,
	0.0254647238733285,
	0.0227880556511908,
	0.0203717790604077,
	0.0181926910596145,
	0.0162295932432360,
	0.0144630941418387,
	0.0128754341866765,
	0.0114503303359322,
	0.0101728378150057,
	0.00902922680011186,
	0.00800687218838523,
	0.00709415486039758,
	0.00628037306181464,
	0.00555566271730628,
	0.00491092564776083,
	0.00433777522517762,
	0.00382858617381395,
	0.00337652538864193,
	0.00297547478958152,
	0.00261995369508530,
	0.00230505051439257,
	0.00202636249613307,
	0.00177994246481535,
	0.00156225163688919,
	0.00137011774112811,
	0.00120069777918906,
	0.00105144485543239,
	0.000920078583646128,
	0.000804558644792605,
	0.000703061126353299,
	0.000613957321970095,
	0.000535794711233811,
	0.000467279874773688,
	0.000407263130174890,
	0.000354724700456040,
	0.000308762228684552,
	0.000268578998820779,
	0.000233472107922766,
	0.000202821534805516,
	0.000176080503619378,
	0.000152766994802780,
	0.000132456257345164,
	0.000114774196621564,
	0.0000993915292610416,
	0.0000860186111205116,
	0.0000744008568854185,
	0.0000643146804615109,
	0.0000555638944463892,
	0.0000479765148133912,
	0.0000414019237006278,
	0.0000357083490382522,
	0.0000307806248038908,
	0.0000265182000840266,
	0.0000228333689341654,
	0.0000196496963539553,
	0.0000169006186225834,
	0.0000145282003166539,
	0.0000124820385512393,
	0.0000107183044508680,
	9.19890566611241e-6,
	7.89075437420041e-6,
	6.76512728089460e-6,
	5.79710594495074e-6,
	4.96508729255373e-6,
	4.25035551717139e-6,
	3.63670770345000e-6,
	3.11012649979137e-6,
	2.65849401629250e-6,
	2.27134186228307e-6,
	1.93963287719169e-6,
	1.65557066379923e-6,
	1.41243351587104e-6,
	1.20442975270958e-6,
	1.02657183986121e-6,
	8.74566995329392e-7,
	7.44722260394541e-7,
	6.33862255545582e-7,
	5.39258025342825e-7,
	4.58565512804405e-7,
	3.89772368391109e-7,
	3.31151972577348e-7,
	2.81223703587451e-7,
	2.38718612981323e-7,
	2.02549784558224e-7,
	1.71786749203399e-7,
	1.45633412099219e-7,
	1.23409021080502e-7,
	1.04531767460094e-7,
	8.85046647687321e-8,
	7.49033977199179e-8,
	6.33658743306062e-8,
	5.35832493603539e-8,
	4.52922178102003e-8,
	3.82684037781748e-8,
	3.23206930422610e-8,
	2.72863777994286e-8,
	2.30269994373198e-8,
	1.94247904820595e-8,
	1.63796304411581e-8,
	1.38064422807221e-8,
	1.16329666668818e-8,
	9.79786000820215e-9,
	8.24906997200364e-9,
	6.94244869879648e-9,
	5.84056956293623e-9,
	4.91171815795476e-9,
	4.12903233557698e-9,
	3.46976969515950e-9,
	2.91468398787199e-9,
	2.44749453802384e-9,
	2.05443505293307e-9,
	1.72387014435469e-9,
	1.44596956306737e-9,
	1.21243159178189e-9,
	1.01624828273784e-9,
	8.51506293255724e-10,
	7.13217989231916e-10,
	5.97178273686798e-10,
	4.99843271868294e-10,
	4.18227580146182e-10,
	3.49817276438660e-10,
	2.92496307733140e-10,
	2.44484226227652e-10,
	2.04283548915435e-10,
	1.70635273863534e-10,
	1.42481306624186e-10,
	1.18932737801671e-10,
	9.92430725748863e-11,
	8.27856490334434e-11,
	6.90345980053579e-11,
	5.75487956079478e-11,
	4.79583435743883e-11,
	3.99531836601083e-11,
	3.32735129630055e-11,
	2.77017183772596e-11,
	2.30555919904645e-11,
	1.91826261797451e-11,
	1.59552184492373e-11,
	1.32666425229607e-11,
	1.10276645918872e-11,
	9.16370253824348e-12,
	7.61244195636034e-12,
	6.32183630823821e-12,
	5.24842997441282e-12,
	4.35595260905192e-12,
	3.61414135533970e-12,
	2.99775435412426e-12,
	2.48574478117179e-12,
	2.06056954190735e-12,
	1.70761087761789e-12,
	1.41469261268532e-12,
	1.17167569925493e-12,
	9.70120179176324e-13,
	8.03002755355921e-13,
	6.64480907032201e-13,
	5.49695947730361e-13,
	4.54608654601190e-13,
	3.75862130571052e-13,
	3.10667427553834e-13,
	2.56708186340823e-13,
	2.12061158957008e-13,
	1.75129990979628e-13,
	1.44590070306053e-13,
	1.19342608376890e-13,
	9.84764210448520e-14,
	8.12361284968988e-14,
	6.69957047626884e-14,
	5.52364839983536e-14,
	4.55288784872365e-14,
	3.75171868260434e-14,
	3.09069739955730e-14,
	2.54545912496319e-14,
	2.09584757642051e-14,
	1.72519300955857e-14,
	1.41971316501794e-14,
	1.16801642038076e-14,
	9.60689839298851e-15,
	7.89957718055663e-15,
	6.49398653148027e-15,
	5.33711172323687e-15,
	4.38519652833446e-15,
	3.60213650413600e-15,
	2.95814927457727e-15,
	2.42867438017647e-15,
	1.99346333303212e-15,
	1.63582721456795e-15,
	1.34201472284939e-15,
	1.10069820297832e-15,
	9.02549036511458e-16,
	7.39886955899583e-16,
	6.06390497499970e-16,
	4.96858003320228e-16,
	4.07010403543137e-16,
	3.33328522514641e-16,
	2.72918903047290e-16,
	2.23403181509686e-16,
	1.82826905742816e-16,
	1.49584399704446e-16,
	1.22356868095946e-16,
	1.00061422004550e-16,
	8.18091101788785e-17,
	6.68703743742468e-17,
	5.46466232309370e-17,
	4.46468473170557e-17,
	3.64683865173660e-17,
	2.97811167122010e-17,
	2.43144513286369e-17,
	1.98466595514452e-17,
	1.61960906400940e-17,
	1.32139661280568e-17,
	1.07784613453433e-17,
	8.78984690826589e-18,
	7.16650138491662e-18,
	5.84163977794677e-18,
	4.76063001400521e-18,
	3.87879232126172e-18,
	3.15959506343337e-18,
	2.57317598320038e-18,
	2.09513046990837e-18,
	1.70551888483764e-18,
	1.38805354722395e-18,
	1.12943303162933e-18,
	9.18797221060242e-19,
	7.47281322095490e-19,
	6.07650960951011e-19,
	4.94003693444398e-19,
	4.01524901266115e-19,
	3.26288213964971e-19,
	2.65092374707276e-19,
	2.15327927385602e-19,
	1.74868299982827e-19,
	1.41980841083036e-19,
	1.15254171584394e-19,
	9.35388736783942e-20,
	7.58990800429806e-20,
	6.15729693405857e-20,
	4.99405370840484e-20,
	4.04973081615272e-20,
	3.28329006413784e-20,
	2.66135496324475e-20,
	2.15678629328980e-20,
	1.74752135068077e-20,
	1.41562828504629e-20,
	1.14653584509271e-20,
	9.28406140589761e-21,
	7.51623982263034e-21,
	6.08381226695129e-21,
	4.92338527497562e-21,
	3.98350139454904e-21,
	3.22240072043320e-21,
	2.60620051521272e-21,
	2.10741515728752e-21,
	1.70375305656048e-21,
	1.37713892323882e-21,
	2.2354265870871718e-27,
)


def rho(x):
	"""Dickman's "rho" function."""
	if x <= 1:
		return 1
	if x < 2:
		return 1 - math.log(x)
	x = (x - 2) * 20
	pos = int(x)

	return rhotab[-1] if pos + 1 >= len(rhotab) else rhotab[pos] + (x - pos) * (rhotab[pos + 1] - rhotab[pos])


def integral(a, b, f, STEPS=20):
	"""Computes the integral of f(x) from a to b."""
	w = b - a
	# assert w >= 0
	if not w:
		return 0
	step = w / STEPS
	return step * sum(f(a + step * (0.5 + i)) for i in range(STEPS))


def p_first_stage(alpha):
	"""Probability of first stage success."""
	return rho(alpha)


def p_second_stage(alpha, beta):
	"""Probability of second stage success."""
	return integral(alpha - beta, alpha - 1, lambda t: rho(t) / (alpha - t))


def primepi(n):
	"""Approximation of the number of primes <= n."""
	return n / (math.log(n) - 1.06)


def n_primes_between(B1, B2):
	"""Returns the number of primes between B1 and B2, inclusive."""
	# assert B2 >= B1
	return primepi(B2) - primepi(B1)


def work_for_bounds(B1, B2, factorB1=1.2, factorB2=1.35):
	"""Returns work for stage-1, stage-2 in the negative (no factor found) case."""
	return B1 * 1.442 * factorB1, n_primes_between(B1, B2) * 0.85 * factorB2


# steps of approx 10%
nice_step = list(chain(range(10, 20), range(20, 40, 2), range(40, 80, 5), range(80, 100, 10)))


def next_nice_number(value):
	"""Use nice round values for bounds."""
	ret = 1
	while value >= nice_step[-1]:
		value //= 10
		ret *= 10
	for n in nice_step:
		if n > value:
			return n * ret
	return None


def pm1(exponent, factoredTo, B1, B2):
	"""Returns the probability of PM1(B1,B2) success for a finding a smooth factor using B1, B2 and already TFed to factoredUpTo."""
	takeAwayBits = log2(exponent) + 1

	SLICE_WIDTH = 0.25
	MIDDLE_SHIFT = log2(1 + 2**SLICE_WIDTH) - 1

	B2 = max(B1, B2)
	bitsB1 = log2(B1)
	bitsB2 = log2(B2)

	alpha = (factoredTo + MIDDLE_SHIFT - takeAwayBits) / bitsB1
	alphaStep = SLICE_WIDTH / bitsB1
	beta = bitsB2 / bitsB1

	sum1 = 0
	sum2 = 0
	invSliceProb = factoredTo / SLICE_WIDTH + 0.5
	p = 1

	while p >= 1e-8:
		p1 = p_first_stage(alpha) / invSliceProb
		p2 = p_second_stage(alpha, beta) / invSliceProb
		sum1 += p1
		sum2 += p2
		p = p1 + p2
		alpha += alphaStep
		invSliceProb += 1

	return -expm1(-sum1), -expm1(-sum2)


def gain(exponent, factoredTo, B1, B2):
	"""Returns tuple (benefit, work) expressed as a ratio of one PRP test."""
	(p1, p2) = pm1(exponent, factoredTo, B1, B2)
	(w1, w2) = work_for_bounds(B1, B2)
	p = p1 + p2
	w = (w1 + (1 - p1 - p2 / 4) * w2) * (1 / exponent)
	return p, w


def walk(exponent, factoredTo):
	"""Optimizes B1 and B2 bounds for a given exponent and factoredTo value."""
	B1 = next_nice_number(exponent // 1000)
	B2 = next_nice_number(exponent // 100)

	# Changes by James Heinrich for mersenne.ca
	# B1mult = (60 - log2(exponent)) / 10000
	# B1 = next_nice_number(int(B1mult * exponent))

	# B2mult = 4 + (log2(exponent) - 20) * 8
	# B2 = next_nice_number(int(B1 * B2mult))
	# End of changes by James Heinrich

	smallB1 = smallB2 = 0
	midB1 = midB2 = 0

	(p, w) = gain(exponent, factoredTo, B1, B2)

	while True:
		stepB1 = next_nice_number(B1) - B1
		stepB2 = next_nice_number(B2) - B2
		(p1, w1) = gain(exponent, factoredTo, B1 + stepB1, B2)
		(p2, w2) = gain(exponent, factoredTo, B1, B2 + stepB2)

		# assert w1 > w and w2 > w and p1 >= p and p2 >= p
		r1 = (p1 - p) / (w1 - w)
		r2 = (p2 - p) / (w2 - w)

		if r1 < 1 and r2 < 1 and not smallB1:
			smallB1 = B1
			smallB2 = B2

		if r1 < 0.5 and r2 < 0.5 and not midB1:
			midB1 = B1
			midB2 = B2

		if r1 < 1 and r2 < 1 and p1 <= w1 and p2 <= w2:
			break

		if r1 > r2:
			B1 += stepB1
			p = p1
			w = w1
		else:
			B2 += stepB2
			p = p2
			w = w2

	if not smallB1:
		if midB1:
			smallB1 = midB1
			smallB2 = midB2
		else:
			smallB1 = B1
			smallB2 = B2

	if not midB1:
		midB1 = B1
		midB2 = B2

	return (smallB1, smallB2), (midB1, midB2), (B1, B2)


# End of Mihai Preda's script

# Python 3.2+
if hasattr(int, "from_bytes"):

	def from_bytes(abytes, byteorder="little"):
		"""Convert a byte sequence to an integer using the specified byte order."""
		return int.from_bytes(abytes, byteorder)

else:

	def from_bytes(abytes, byteorder="little"):
		"""Convert a bytes sequence to an integer with the specified byte order."""
		if byteorder == "big":
			abytes = reversed(abytes)
		return sum(b << i * 8 for i, b in enumerate(bytearray(abytes)))


def unpack(aformat, file, noraise=False):
	"""Unpacks binary data from a file according to the specified format."""
	size = struct.calcsize(aformat)
	buffer = file.read(size)
	if len(buffer) != size:
		if noraise and not buffer:
			return None
		raise EOFError
	return struct.unpack(aformat, buffer)


def read_residue_mlucas(file, nbytes):
	"""Reads and unpacks residue data from a file at a given byte offset."""
	file.seek(nbytes, 1)  # os.SEEK_CUR

	res64, res35m1, res36m1 = unpack("<Q5s5s", file)
	# res35m1 = from_bytes(res35m1)
	# res36m1 = from_bytes(res36m1)
	return res64, res35m1, res36m1


def parse_work_unit_mlucas(adapter, filename, exponent, astage):
	"""Parses a Mlucas work unit file and extract information."""
	counter = 0
	pct_complete = None
	fftlen = None

	try:
		with open(filename, "rb") as f:
			t, m, tmp = unpack("<BB8s", f)
			nsquares = from_bytes(tmp)

			p = 1 << exponent if m == MODULUS_TYPE_FERMAT else exponent

			nbytes = (p + 7) // 8 if m == MODULUS_TYPE_MERSENNE else (p >> 3) + 1 if m == MODULUS_TYPE_FERMAT else 0

			_res64, _res35m1, _res36m1 = read_residue_mlucas(f, nbytes)

			result = unpack("<3sQ", f, True)
			kblocks = _res_shift = None
			if result is not None:
				kblocks, _res_shift = result
				kblocks = from_bytes(kblocks)

			if t == TEST_TYPE_PRP or (t == TEST_TYPE_PRIMALITY and m == MODULUS_TYPE_FERMAT):
				(_prp_base,) = unpack("<I", f)

				_i1, _i2, _i3 = read_residue_mlucas(f, nbytes)

				(_gcheck_shift,) = unpack("<Q", f)

			result = unpack("<II", f, True)
			if result is not None:
				_nerr_roe, _nerr_gcheck = result

			if t == TEST_TYPE_PRIMALITY:
				if m == MODULUS_TYPE_MERSENNE:
					counter = nsquares
					stage = "LL"
					pct_complete = nsquares / (p - 2)
			elif t == TEST_TYPE_PRP:
				counter = nsquares
				stage = "PRP"
				pct_complete = nsquares / p
			elif t == TEST_TYPE_PM1:
				if astage == 1:
					counter = nsquares
				# elif astage == 2:
				#     interim_C = from_bytes(tmp[:-1])
				#     _psmall = from_bytes(tmp[-1:])
				stage = "S{0}".format(astage)
			else:
				adapter.debug("savefile with unknown TEST_TYPE = %s", t)
				return None

			if m != MODULUS_TYPE_MERSENNE:
				adapter.debug("savefile with unknown MODULUS_TYPE = %s", m)
				return None

			if kblocks is not None:
				fftlen = kblocks << 10
	except EOFError:
		return None
	except (IOError, OSError) as e:
		logging.exception("Error reading %r file: %s", filename, e, exc_info=options.debug)
		return None

	return counter, stage, pct_complete, fftlen


def parse_work_unit_cudalucas(adapter, filename, p):
	"""Parses a CUDALucas work unit file and extract information."""
	end = (p + 31) // 32

	try:
		with open(filename, "rb") as f:
			f.seek(end * 4)

			q, n, j, _offset, total_time, _time_adj, _iter_adj, _, magic_number, _checksum = unpack("=IIIIIIIIII", f)
			if p != q:
				adapter.debug("Error: Expecting the exponent %s, but found %s", p, q)
				return None
			if magic_number:
				adapter.debug("Error: savefile with unknown magic_number = %s", magic_number)
				return None
			total_time <<= 15
			# _time_adj <<= 15

			counter = j
			fftlen = n
			avg_msec_per_iter = (total_time / j) / 1000
			stage = "LL"
			pct_complete = j / (q - 2)
	except EOFError:
		return None
	except (IOError, OSError) as e:
		logging.exception("Error reading %r file: %s", filename, e, exc_info=options.debug)
		return None

	return counter, avg_msec_per_iter, stage, pct_complete, fftlen


# GpuOwl/PRPLL headers


# Exponent, iteration, 0, hash
LL_v1_RE = re.compile(br"^OWL LL (1) (\d+) (\d+) 0 ([\da-f]+)$")

# E, k, CRC
LL_v1a_RE = re.compile(br"^OWL LL (1) E=(\d+) k=(\d+) CRC=(\d+)$")

LL_v13_RE = re.compile(br"^OWL LL (13) N=1\*2\^(\d+)-1 k=(\d+) time=(\d+(?:\.\d+)?)$")

# Exponent, iteration, block-size, res64
PRP_v9_RE = re.compile(br"^OWL PRP (9) (\d+) (\d+) (\d+) ([\da-f]{16})$")

# E, k, block-size, res64, nErrors
PRP_v10_RE = re.compile(br"^OWL PRP (10) (\d+) (\d+) (\d+) ([\da-f]{16}) (\d+)$")

# Exponent, iteration, block-size, res64, nErrors, B1, nBits, start, nextK, crc
PRP_v11_RE = re.compile(br"^OWL PRP (11) (\d+) (\d+) (\d+) ([\da-f]{16}) (\d+)(?: (\d+) (\d+) (\d+) (\d+) (\d+))?$")

# E, k, block-size, res64, nErrors, CRC
PRP_v12_RE = re.compile(br"^OWL PRP (12) (\d+) (\d+) (\d+) ([\da-f]{16}) (\d+) (\d+)$")

PRP_v13_RE = re.compile(br"^OWL PRP (13) N=1\*2\^(\d+)-1 k=(\d+) block=(\d+) res64=([\da-f]{16}) err=(\d+) time=(\d+(?:\.\d+)?)$")

# Exponent, B1, iteration, nBits
P1_v1_RE = re.compile(br"^OWL PM?1 (1) (\d+) (\d+) (\d+) (\d+)$")

# E, B1, k, nextK, CRC
P1_v2_RE = re.compile(br"^OWL P1 (2) (\d+) (\d+) (\d+) (\d+) (\d+)$")

P1_v3_RE = re.compile(br"^OWL P1 (3) E=(\d+) B1=(\d+) k=(\d+)(?: block=(\d+))?$")

# E, B1, CRC
P1Final_v1_RE = re.compile(br"^OWL P1F (1) (\d+) (\d+) (\d+)$")

# Exponent, B1, B2, nWords, kDone
P2_v1_RE = re.compile(br"^OWL P2 (1) (\d+) (\d+) (\d+) (\d+) 2880 (\d+)$")

# E, B1, B2, CRC
P2_v2_RE = re.compile(br"^OWL P2 (2) (\d+) (\d+) (\d+)(?: (\d+))?$")

# E, B1, B2, D, nBuf, nextBlock
P2_v3_RE = re.compile(br"^OWL P2 (3) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)$")


def parse_work_unit_gpuowl(adapter, filename, p):
	"""Parses a GpuOwl work unit file and extract information."""
	counter = 0
	avg_msec_per_iter = None
	pct_complete = None
	buffs = bits = 0

	try:
		with open(filename, "rb") as f:
			header = f.readline().rstrip(b"\n")
	except (IOError, OSError) as e:
		logging.exception("Error reading %r file: %s", filename, e, exc_info=options.debug)
		return None

	if not header.startswith(b"OWL "):
		return None

	if header.startswith(b"OWL LL "):
		ll_v13 = LL_v13_RE.match(header)
		ll_v1a = LL_v1a_RE.match(header)
		ll_v1 = LL_v1_RE.match(header)

		elapsed = None

		if ll_v13:
			_version, exponent, iteration, elapsed = ll_v13.groups()
		elif ll_v1a:
			_version, exponent, iteration, _crc = ll_v1a.groups()
		elif ll_v1:
			_version, exponent, iteration, _ahash = ll_v1.groups()
		else:
			adapter.debug("LL savefile with unknown version: %r", header)
			return None

		counter = int(iteration)
		stage = "LL"
		pct_complete = counter / (p - 2)
		if elapsed:
			avg_msec_per_iter = (float(elapsed) / counter) * 1000
	elif header.startswith(b"OWL PRP "):
		prp_v13 = PRP_v13_RE.match(header)
		prp_v12 = PRP_v12_RE.match(header)
		prp_v11 = PRP_v11_RE.match(header)
		prp_v10 = PRP_v10_RE.match(header)
		prp_v9 = PRP_v9_RE.match(header)

		elapsed = None

		if prp_v13:
			_version, exponent, iteration, _block_size, _res64, _nErrors, elapsed = prp_v13.groups()
		elif prp_v12:
			_version, exponent, iteration, _block_size, _res64, _nErrors, _crc = prp_v12.groups()
		elif prp_v11:
			_version, exponent, iteration, _block_size, _res64, _nErrors, _B1, _nBits, _start, _nextK, _crc = prp_v11.groups()
		elif prp_v10:
			_version, exponent, iteration, _block_size, _res64, _nErrors = prp_v10.groups()
		elif prp_v9:
			_version, exponent, iteration, _block_size, _res64 = prp_v9.groups()
		else:
			adapter.debug("PRP savefile with unknown version: %r", header)
			return None

		counter = int(iteration)
		stage = "PRP"
		pct_complete = counter / p
		if elapsed:
			avg_msec_per_iter = (float(elapsed) / counter) * 1000
	elif header.startswith((b"OWL PM1 ", b"OWL P1 ", b"OWL P1F ")):
		p1_v3 = P1_v3_RE.match(header)
		p1_v2 = P1_v2_RE.match(header)
		p1_v1 = P1_v1_RE.match(header)
		p1final_v1 = P1Final_v1_RE.match(header)

		if p1_v3:
			_version, exponent, _B1, iteration, _block = p1_v3.groups()
			counter = int(iteration)
		elif p1_v2:
			_version, exponent, _B1, iteration, _nextK, _crc = p1_v2.groups()
			counter = int(iteration)
		elif p1_v1:
			_version, exponent, _B1, iteration, nBits = p1_v1.groups()
			counter = int(iteration)
			bits = int(nBits)
			pct_complete = counter / bits
		elif p1final_v1:
			_version, exponent, _B1, _crc = p1final_v1.groups()
			pct_complete = 1.0
		else:
			adapter.debug("P-1 stage 1 savefile with unknown version: %r", header)
			return None

		# B_done = int(B1)
		stage = "S1"
	elif header.startswith(b"OWL P2 "):
		p2_v3 = P2_v3_RE.match(header)
		p2_v2 = P2_v2_RE.match(header)
		p2_v1 = P2_v1_RE.match(header)

		if p2_v3:
			_version, exponent, _B1, _B2, _D, _nBuf, nextBlock = p2_v3.groups()
			if int(nextBlock) == 0xFFFFFFFF:  # (1 << 32) - 1
				pct_complete = 1.0
		elif p2_v2:
			_version, exponent, _B1, _B2, _crc = p2_v2.groups()
		elif p2_v1:
			_version, exponent, _B1, _B2, _nWords, kDone = p2_v1.groups()
			counter = int(kDone)
			buffs = 2880
			pct_complete = counter / buffs
		else:
			adapter.debug("P-1 stage 2 savefile with unknown version: %r", header)
			return None

		# B_done = int(B1)
		# C_done = int(B2)
		stage = "S2"
	else:
		adapter.debug("Error: Unknown save/checkpoint file header: %r", header)
		return None

	if p != int(exponent):
		return None

	return counter, avg_msec_per_iter, stage, pct_complete, bits, buffs


def calculate_k(exp, bits):
	"""Calculate the value of k based on the given exponent and bit length."""
	tmp_low = 1 << (bits - 1)
	tmp_low -= 1
	k = tmp_low // exp

	if k == 0:
		k = 1
	return k


def class_needed(exp, k_min, c, more_classes):
	"""Determines if a class is needed based on given parameters and conditions."""
	if (
		(2 * (exp % 8) * ((k_min + c) % 8)) % 8 != 2
		and ((2 * (exp % 8) * ((k_min + c) % 8)) % 8 != 4)
		and ((2 * (exp % 3) * ((k_min + c) % 3)) % 3 != 2)
		and ((2 * (exp % 5) * ((k_min + c) % 5)) % 5 != 4)
		and ((2 * (exp % 7) * ((k_min + c) % 7)) % 7 != 6)
	):
		if not more_classes or (2 * (exp % 11) * ((k_min + c) % 11)) % 11 != 10:
			return True

	return False


def pct_complete_mfakt(exp, bits, num_classes, cur_class):
	"""Calculate the percentage of completion for the exponent based on the current class."""
	# Lines of code with comments below are taken from mfaktc.c

	cur_class += 1  # the checkpoint contains the last complete processed class!

	if num_classes in {4620, 420}:
		if num_classes == 4620:
			more_classes = True
			max_class_number = 960
		elif num_classes == 420:
			more_classes = False
			max_class_number = 96

		k_min = calculate_k(exp, bits)
		k_min -= k_min % num_classes  # k_min is now 0 mod num_classes

		class_counter = sum(1 for i in range(cur_class) if class_needed(exp, k_min, i, more_classes))

		return class_counter / max_class_number

	# This should never happen
	return cur_class / num_classes


def tf_ghd_credit(exp, bit_min, bit_max):
	"""Calculate the GHz-days credit for a given exponent and bit range."""
	ghzdays = sum(
		(0.011160 if i <= 62 else 0.017832 if i <= 64 else 0.016968) * 2 ** (i - 48) for i in range(bit_min + 1, bit_max + 1)
	)
	ghzdays *= 1680 / exp
	return ghzdays


# "%s%u %d %d %d %s: %d %d %s %llu %08X", NAME_NUMBERS, exp, bit_min, bit_max, NUM_CLASSES, MFAKTC_VERSION, cur_class, num_factors, strlen(factors_string) ? factors_string : "0", bit_level_time, i
MFAKTC_TF_RE = re.compile(br'^M(\d+) (\d+) (\d+) (\d+) ([^\s:]+): (\d+) (\d+) (0|"\d+"(?:,"\d+")*) (\d+) ([\dA-F]{8})$')


def parse_work_unit_mfaktc(filename, p):
	"""Parses a mfaktc work unit file, extracting important information."""
	try:
		with open(filename, "rb") as f:
			header = f.readline().rstrip(b"\n")
	except (IOError, OSError) as e:
		logging.exception("Error reading %r file: %s", filename, e, exc_info=options.debug)
		return None

	mfaktc_tf = MFAKTC_TF_RE.match(header)

	if mfaktc_tf:
		exp, bit_min, bit_max, num_classes, _version, cur_class, _num_factors, _factors_string, bit_level_time, _i = (
			mfaktc_tf.groups()
		)
	else:
		return None

	n = int(exp)
	bits = int(bit_min)
	ms_elapsed = int(bit_level_time)

	if p != n:
		return None

	pct_complete = pct_complete_mfakt(n, bits, int(num_classes), int(cur_class))
	assignment_ghd = tf_ghd_credit(n, bits, int(bit_max))
	counter = pct_complete * assignment_ghd
	avg_msec_per_iter = ms_elapsed / counter if ms_elapsed else None

	return counter, avg_msec_per_iter, pct_complete


# "%u %d %d %d %s: %d %d %s %llu %08X\n", exp, bit_min, bit_max, mystuff.num_classes, MFAKTO_VERSION, cur_class, num_factors, strlen(factors_string) ? factors_string : "0", bit_level_time, i
MFAKTO_TF_RE = re.compile(br'^(\d+) (\d+) (\d+) (\d+) (mfakto [^\s:]+): (\d+) (\d+) (0|"\d+"(?:,"\d+")*) (\d+) ([\dA-F]{8})$')


def parse_work_unit_mfakto(filename, p):
	"""Parses a mfakto work unit file, extracting important information."""
	try:
		with open(filename, "rb") as f:
			header = f.readline().rstrip(b"\n")
	except (IOError, OSError) as e:
		logging.exception("Error reading %r file: %s", filename, e, exc_info=options.debug)
		return None

	mfakto_tf = MFAKTO_TF_RE.match(header)

	if mfakto_tf:
		exp, bit_min, bit_max, num_classes, _version, cur_class, _num_factors, _factors_string, bit_level_time, _i = (
			mfakto_tf.groups()
		)
	else:
		return None

	n = int(exp)
	bits = int(bit_min)
	ms_elapsed = int(bit_level_time)

	if p != n:
		return None

	pct_complete = pct_complete_mfakt(n, bits, int(num_classes), int(cur_class))
	assignment_ghd = tf_ghd_credit(n, bits, int(bit_max))
	counter = pct_complete * assignment_ghd
	avg_msec_per_iter = ms_elapsed / counter if ms_elapsed else None

	return counter, avg_msec_per_iter, pct_complete


def get_stages_mfaktx_ini(adapter, adir):
	"""Retrieve the number of stages from the mfaktc.ini or mfakto.ini configuration file."""
	stages = 1
	ini_file = os.path.join(adir, "mfaktc.ini" if options.mfaktc else "mfakto.ini")
	if not os.path.exists(ini_file):
		adapter.debug("Configuration file %r does not exist", ini_file)
		return stages
	config = ConfigParser()
	try:
		with io.open(ini_file) as file:
			if hasattr(config, "read_file"):  # Python 3.2+
				config.read_file(chain(("[default]",), file))
			else:
				with io.StringIO() as stream:
					stream.write("[default]\n")
					stream.write(file.read())
					stream.seek(0)
					config.readfp(stream)
	except ConfigParserError as e:
		adapter.exception("Error reading %r configuration file: %s", ini_file, e, exc_info=options.debug)
	if config.has_option("default", "Stages"):
		stages = config.getint("default", "Stages")
	return stages


MLUCAS_RE = re.compile(r"^p([0-9]+)(?:\.s([12]))?$")


def parse_stat_file(adapter, adir, p):
	"""Parse the Mlucas stat file for the progress of the assignment."""
	# Mlucas
	savefiles = []
	for entry in glob.iglob(os.path.join(adir, "p{0}*".format(p))):
		match = MLUCAS_RE.match(os.path.basename(entry))
		if match:
			astage = match.group(2) and int(match.group(2))
			savefiles.append((1 if astage is None else astage, entry))
	iteration = 0
	stage = pct_complete = None
	fftlen = None
	for astage, savefile in sorted(savefiles, reverse=True):
		result = parse_work_unit_mlucas(adapter, savefile, p, astage)
		if result is not None:
			iteration, stage, pct_complete, fftlen = result
			break
	else:
		adapter.debug("No save/checkpoint files found for the exponent %s", p)

	statfile = os.path.join(adir, "p{0}.stat".format(p))
	if not os.path.exists(statfile):
		adapter.debug("stat file %r does not exist", statfile)
		return iteration, None, stage, pct_complete, fftlen, 0, 0

	w = readonly_list_file(statfile)  # appended line by line, no lock needed
	found = 0
	regex = re.compile(
		r"(Iter#|S1|S2)(?: bit| at q)? = ([0-9]+) \[ ?([0-9]+\.[0-9]+)% complete\] .*\[ *([0-9]+\.[0-9]+) (m?sec)/iter\]"
	)
	fft_regex = re.compile(r"FFT length [0-9]{3,}K = ([0-9]{6,})")
	s2_regex = re.compile(r"Stage 2 q0 = ([0-9]+)")
	list_msec_per_iter = []
	s2 = bits = 0
	# get the 5 most recent Iter line
	for line in reversed(list(w)):
		res = regex.search(line)
		fft_res = fft_regex.search(line)
		s2_res = s2_regex.search(line)
		if res and found < 5:
			astage = res.group(1)
			# keep the last iteration to compute the percent of progress
			if not found:
				iteration = int(res.group(2))
				percent = float(res.group(3))
				if astage in {"S1", "S2"}:
					stage = astage
				if astage == "S1":
					bits = int(iteration / (percent / 100))
				elif astage == "S2":
					s2 = iteration
			if (not bits or astage == "S1") and (not s2 or astage == "S2"):
				msec_per_iter = float(res.group(4))
				if res.group(5) == "sec":
					msec_per_iter *= 1000
				list_msec_per_iter.append(msec_per_iter)
			found += 1
		elif s2 and s2_res:
			s2 = int((iteration - int(s2_res.group(1))) / (percent / 100) / 20)
			iteration = int(s2 * (percent / 100))
		elif fft_res and not fftlen:
			fftlen = int(fft_res.group(1))
		if found == 5 and not s2 and fftlen:
			break
	if not found:
		# iteration is 0, but don't know the estimated speed yet
		return iteration, None, stage, pct_complete, fftlen, bits, s2
	# take the median of the last grepped lines
	msec_per_iter = median_low(list_msec_per_iter)
	return iteration, msec_per_iter, stage, pct_complete, fftlen, bits, s2


def parse_cuda_output_file(adapter, adir, p):
	"""Parse the CUDALucas output file for the progress of the assignment."""
	# CUDALucas
	savefile = os.path.join(adir, "c{0}".format(p))
	iteration = 0
	avg_msec_per_iter = None
	stage = pct_complete = None
	fftlen = None
	buffs = bits = 0
	if os.path.exists(savefile):
		result = parse_work_unit_cudalucas(adapter, savefile, p)
		if result is not None:
			iteration, avg_msec_per_iter, stage, pct_complete, fftlen = result
	else:
		adapter.debug("Save/Checkpoint file %r does not exist", savefile)

	return iteration, avg_msec_per_iter, stage, pct_complete, fftlen, bits, buffs


GPUOWL_RE = re.compile(r"^(?:(?:[0-9]+)(?:-([0-9]+)\.(ll|prp|p1final|p2)|(?:-[0-9]+-([0-9]+))?\.p1|\.(?:(ll|p[12])\.)?owl))$")


def parse_gpu_log_file(adapter, adir, p):
	"""Parse the GpuOwl log file for the progress of the assignment."""
	# GpuOwl
	savefiles = []
	for entry in glob.iglob(os.path.join(adir, "*{0}".format(p), "{0}*.*".format(p))):
		match = GPUOWL_RE.match(os.path.basename(entry))
		if match:
			savefiles.append((
				2
				if match.group(2) == "prp" or (match.group(2) or match.group(4)) == "ll"
				else 1
				if (match.group(2) or match.group(4)) == "p2"
				else 0,
				tuple(map(int, os.path.basename(entry).split(".", 1)[0].split("-"))),
				entry,
			))
	iteration = 0
	stage = pct_complete = None
	fftlen = None
	buffs = bits = 0
	for _, _, savefile in sorted(savefiles, reverse=True):
		result = parse_work_unit_gpuowl(adapter, savefile, p)
		if result is not None:
			iteration, avg_msec_per_iter, stage, pct_complete, bits, buffs = result
			if avg_msec_per_iter is not None:
				return iteration, avg_msec_per_iter, stage, pct_complete, fftlen, bits, buffs
			break
	else:
		adapter.debug("No save/checkpoint files found for the exponent %s", p)

	# appended line by line, no lock needed
	logfile = os.path.join(adir, "gpuowl.log")
	if not os.path.exists(logfile):
		adapter.debug("Log file %r does not exist", logfile)
		return iteration, None, stage, pct_complete, fftlen, 0, 0

	w = readonly_list_file(logfile)
	found = 0
	regex = re.compile(r"([0-9]{6,}) (LL|P1|OK|EE)? +([0-9]{4,})")
	aregex = re.compile(r"([0-9]{6,})P1 +([0-9]+\.[0-9]+)% ([KE])? +[0-9a-f]{16} +([0-9]+)")
	us_per_regex = re.compile(r"\b([0-9]+) us/it;?")
	fft_regex = re.compile(r"\b([0-9]{6,}) FFT: ([0-9]+(?:\.[0-9]+)?[KM])\b")
	p1_bits_regex = re.compile(r"\b[0-9]{6,} P1(?: B1=[0-9]+, B2=[0-9]+;|\([0-9]+(?:\.[0-9])?M?\)) ([0-9]+) bits;?\b")
	ap1_bits_regex = re.compile(r"\b[0-9]{6,}P1 +[0-9]+\.[0-9]+% @([0-9]+)/([0-9]+) B1\([0-9]+\)")
	p2_blocks_regex = re.compile(r"[0-9]{6,} P2\([0-9]+(?:\.[0-9])?M?,[0-9]+(?:\.[0-9])?M?\) ([0-9]+) blocks: ([0-9]+) - ([0-9]+);")
	p1_regex = re.compile(r"\| P1\([0-9]+(?:\.[0-9])?M?\)")
	p2_regex = re.compile(r"[0-9]{6,} P2(?: ([0-9]+)/([0-9]+)|\([0-9]+(?:\.[0-9])?M?,[0-9]+(?:\.[0-9])?M?\) OK @([0-9]+)):")
	list_usec_per_iter = []
	p1 = p2 = False
	# get the 5 most recent Iter line
	for line in reversed(list(w)):
		res = regex.search(line)
		ares = aregex.search(line)
		us_res = us_per_regex.search(line)
		fft_res = fft_regex.search(line)
		p1_bits_res = p1_bits_regex.search(line)
		ap1_bits_res = ap1_bits_regex.search(line)
		blocks_res = p2_blocks_regex.search(line)
		p2_res = p2_regex.search(line)
		if res or ares:
			num = int(res.group(1) if res else ares.group(1))
			if num != p:
				if not found:
					adapter.debug("Looking for the exponent %s, but found %s", p, num)
				break
		if p2_res:
			found += 1
			if found == 1:
				if p2_res.group(3):
					iteration = int(p2_res.group(3))
					p2 = True
				else:
					iteration = int(p2_res.group(1))
					buffs = int(p2_res.group(2))  # 2880
				stage = "S2"
		elif res and us_res and found < 20:
			found += 1
			aiteration = int(res.group(3))
			# keep the last iteration to compute the percent of progress
			if found == 1:
				iteration = aiteration
				p1 = res.group(2) == "P1"
				if p1:
					stage = "S1"
			elif aiteration > iteration:
				break
			if not p1 and not (p2 or buffs):
				p1_res = p1_regex.search(line)
				p1 = res.group(2) == "OK" and bool(p1_res)
				if p1:
					stage = "S1"
			if len(list_usec_per_iter) < 5:
				list_usec_per_iter.append(int(us_res.group(1)))
		elif ares and found < 20:
			found += 1
			apercent = float(ares.group(2))
			if found == 1:
				percent = apercent
				p1 = True
				stage = "S1"
			elif apercent > percent:
				break
			if len(list_usec_per_iter) < 5:
				list_usec_per_iter.append(int(ares.group(4)))
		elif p2 and blocks_res:
			if not buffs:
				buffs = int(blocks_res.group(1))
				iteration -= int(blocks_res.group(2))
		elif p1 and (p1_bits_res or ap1_bits_res):
			if not bits:
				if p1_bits_res:
					bits = int(p1_bits_res.group(1))
					iteration = min(iteration, bits)
				else:
					bits = int(ap1_bits_res.group(2))
					iteration = int(bits * (percent / 100))
		elif fft_res and not fftlen:
			if int(fft_res.group(1)) != p:
				break
			fftlen = inputunit(fft_res.group(2))
		if (buffs or (found == 20 and not p2 and (not p1 or bits))) and fftlen:
			break
	if not found:
		# iteration is 0, but don't know the estimated speed yet
		return iteration, None, stage, pct_complete, fftlen, bits, buffs
	# take the median of the last grepped lines
	msec_per_iter = median_low(list_usec_per_iter) / 1000 if list_usec_per_iter else None
	return iteration, msec_per_iter, stage, pct_complete, fftlen, bits, buffs


def parse_mfaktc_output_file(adir, p):
	"""Parse the mfaktc output file for the progress of the assignment."""
	savefile = os.path.join(adir, "M{0}.ckp".format(p))
	iteration = 0
	avg_msec_per_iter = None
	stage = pct_complete = None
	if os.path.exists(savefile):
		result = parse_work_unit_mfaktc(savefile, p)
		if result is not None:
			iteration, avg_msec_per_iter, pct_complete = result
	# else:
	#     adapter.debug("Checkpoint file %r does not exist", savefile)

	return iteration, avg_msec_per_iter, stage, pct_complete, None, 0, 0


def parse_mfakto_output_file(adir, p):
	"""Parse the mfakto output file for the progress of the assignment."""
	savefile = os.path.join(adir, "M{0}.ckp".format(p))
	iteration = 0
	avg_msec_per_iter = None
	stage = pct_complete = None
	if os.path.exists(savefile):
		result = parse_work_unit_mfakto(savefile, p)
		if result is not None:
			iteration, avg_msec_per_iter, pct_complete = result
	# else:
	#     adapter.debug("Checkpoint file %r does not exist", savefile)

	return iteration, avg_msec_per_iter, stage, pct_complete, None, 0, 0


def get_progress_assignment(adapter, adir, assignment):
	"""Retrieve the progress of an assignment."""
	if not assignment:
		return None
	if options.gpuowl:  # GpuOwl
		result = parse_gpu_log_file(adapter, adir, assignment.n)
	elif options.cudalucas:  # CUDALucas
		result = parse_cuda_output_file(adapter, adir, assignment.n)
	elif options.mfaktc:  # mfaktc
		result = parse_mfaktc_output_file(adir, assignment.n)
	elif options.mfakto:  # mfakto
		result = parse_mfakto_output_file(adir, assignment.n)
	else:  # Mlucas
		result = parse_stat_file(adapter, adir, assignment.n)
	return result


def compute_progress(assignment, iteration, msec_per_iter, p, bits, s2):
	"""Calculate the progress percentage and estimated time left for a given assignment."""
	percent = iteration / (
		s2
		or bits
		or (
			assignment.n
			if assignment.work_type == PRIMENET.WORK_TYPE_PRP
			else assignment.cert_squarings
			if assignment.work_type == PRIMENET.WORK_TYPE_CERT
			else tf_ghd_credit(assignment.n, int(assignment.sieve_depth), int(assignment.factor_to))
			if assignment.work_type == PRIMENET.WORK_TYPE_FACTOR
			else assignment.n - 2
		)
	)
	if msec_per_iter is None:
		return percent, None, msec_per_iter
	if assignment.n != p and assignment.work_type != PRIMENET.WORK_TYPE_FACTOR:
		msec_per_iter *= assignment.n * log2(assignment.n) * log2(log2(assignment.n)) / (p * log2(p) * log2(log2(p)))
	if bits:
		time_left = msec_per_iter * (bits - iteration)
		# 1.5 suggested by EWM for Mlucas v20.0 and 1.13-1.275 for v20.1
		time_left += msec_per_iter * bits * 1.2
		if assignment.work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK, PRIMENET.WORK_TYPE_PRP}:
			time_left += msec_per_iter * (assignment.n if assignment.work_type == PRIMENET.WORK_TYPE_PRP else assignment.n - 2)
	elif s2:
		time_left = msec_per_iter * (s2 - iteration) if not options.gpuowl else options.timeout
		if assignment.work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK, PRIMENET.WORK_TYPE_PRP}:
			time_left += msec_per_iter * (assignment.n if assignment.work_type == PRIMENET.WORK_TYPE_PRP else assignment.n - 2)
	elif assignment.work_type in {PRIMENET.WORK_TYPE_PMINUS1, PRIMENET.WORK_TYPE_PFACTOR}:
		# assume P-1 time is 1.75% of a PRP test (from Prime95)
		time_left = msec_per_iter * assignment.n * 0.0175
	elif assignment.work_type == PRIMENET.WORK_TYPE_FACTOR:
		time_left = msec_per_iter * (
			tf_ghd_credit(assignment.n, int(assignment.sieve_depth), int(assignment.factor_to)) - iteration
		)
	else:
		time_left = msec_per_iter * (
			(
				assignment.n
				if assignment.work_type == PRIMENET.WORK_TYPE_PRP
				else assignment.cert_squarings
				if assignment.work_type == PRIMENET.WORK_TYPE_CERT
				else assignment.n - 2
			)
			- iteration
		)
	rolling_average = config.getint(SEC.Internals, "RollingAverage") if config.has_option(SEC.Internals, "RollingAverage") else 1000
	rolling_average = min(4000, max(10, rolling_average))
	time_left *= (24 / options.cpu_hours) * (1000 / rolling_average)
	return percent, time_left / 1000, msec_per_iter


def work_estimate(adapter, adir, cpu_num, assignment):
	"""Estimate the remaining work time for a given assignment."""
	section = "Worker #{0}".format(cpu_num + 1) if options.num_workers > 1 else SEC.Internals
	msec_per_iter = p = None
	if config.has_option(section, "msec_per_iter") and config.has_option(section, "exponent"):
		msec_per_iter = config.getfloat(section, "msec_per_iter")
		p = config.getint(section, "exponent")
	iteration, _, _, _, _, bits, s2 = get_progress_assignment(adapter, adir, assignment)
	_, time_left, _ = compute_progress(assignment, iteration, msec_per_iter, p, bits, s2)
	return time_left


def string_to_hash(astr):
	"""Converts a string to a hash value using a modified MD5 algorithm."""
	md5_hash = md5(astr.encode("utf-8")).hexdigest()

	ahash = 0
	for i in range(0, len(md5_hash), 8):
		val = 0
		for j in range(8):
			temp = int(md5_hash[i + j], 16)
			# The 32 results from a bug in Prime95/MPrime
			val = (val << 4) + temp + (32 if temp >= 10 else 0)
		ahash += val

	return ahash & 0x7FFFFFFF


def rolling_average_work_unit_complete(adapter, adir, cpu_num, tasks, assignment):
	"""Updates rolling average work unit completion time and hash based on the next assignment."""
	ahash = config.getint(SEC.Internals, "RollingHash") if config.has_option(SEC.Internals, "RollingHash") else 0
	time_to_complete = (
		config.getint(SEC.Internals, "RollingCompleteTime") if config.has_option(SEC.Internals, "RollingCompleteTime") else 0
	)

	next_assignment = next((task for task in tasks if isinstance(task, Assignment)), None)
	if next_assignment is not None:
		ahash -= string_to_hash(exponent_to_str(assignment))
		ahash += string_to_hash(exponent_to_str(next_assignment))
		ahash &= 0x7FFFFFFF

		time_left = work_estimate(adapter, adir, cpu_num, next_assignment)
		if time_left is None:
			return
		time_to_complete += time_left

	config.set(SEC.Internals, "RollingHash", str(ahash))
	config.set(SEC.Internals, "RollingCompleteTime", str(int(time_to_complete)))


def adjust_rolling_average(dirs):
	"""Adjusts the 30-day rolling average based on the current work assignments."""
	current_time = time.time()
	ahash = 0
	time_to_complete = 0
	for i, adir in enumerate(dirs):
		adapter = logging.LoggerAdapter(logger, {"cpu_num": i} if options.dirs else None)
		cpu_num = i if options.dirs else options.cpu
		tasks = read_workfile(adapter, adir)
		assignment = next((assignment for assignment in tasks if isinstance(assignment, Assignment)), None)
		if assignment is None:
			continue
		ahash += string_to_hash(exponent_to_str(assignment))
		time_left = work_estimate(adapter, adir, cpu_num, assignment)
		if time_left is None:
			return
		time_to_complete += time_left

	rolling_hash = config.getint(SEC.Internals, "RollingHash") if config.has_option(SEC.Internals, "RollingHash") else 0
	start_time = config.getint(SEC.Internals, "RollingStartTime") if config.has_option(SEC.Internals, "RollingStartTime") else 0
	complete_time = (
		config.getint(SEC.Internals, "RollingCompleteTime") if config.has_option(SEC.Internals, "RollingCompleteTime") else 0
	)
	rolling_average = config.getint(SEC.Internals, "RollingAverage") if config.has_option(SEC.Internals, "RollingAverage") else 1000

	ahash &= 0x7FFFFFFF
	if ahash == rolling_hash:
		delta = current_time - start_time
		if (
			start_time
			and current_time > start_time
			and timedelta(seconds=delta) <= timedelta(days=30)
			and complete_time > time_to_complete
		):
			arolling_average = (complete_time - time_to_complete) / options.num_workers / delta * rolling_average
			if arolling_average <= 50000:
				arolling_average = min(2 * rolling_average, max(rolling_average // 2, arolling_average))

				pct = delta / (30 * 24 * 60 * 60)
				rolling_average = int((1.0 - pct) * rolling_average + pct * arolling_average + 0.5)
				logging.info(
					"Updating 30-day rolling average to %s (using %s of %s)", rolling_average, format(pct, "%"), arolling_average
				)

				rolling_average = min(4000, max(20, rolling_average))
			else:
				logging.debug("The rolling average is too large (%s > 50000), not updating 30-day value", arolling_average)
		else:
			logging.debug("The workfile was modified, not updating rolling average")

	config.set(SEC.Internals, "RollingHash", str(ahash))
	config.set(SEC.Internals, "RollingStartTime", str(int(current_time)))
	config.set(SEC.Internals, "RollingCompleteTime", str(int(time_to_complete)))
	config.set(SEC.Internals, "RollingAverage", str(rolling_average))


def output_status(dirs, cpu_num=None):
	"""Outputs the status of queued work and expected completion dates for given directories."""
	logging.info("Below is a report on the work you have queued and any expected completion dates.")
	ll_and_prp_cnt = 0
	prob = 0.0
	adapter = logging.LoggerAdapter(logger, None)
	for i, adir in enumerate(dirs):
		if options.status and options.num_workers > 1:
			logging.info("[Worker #%s]", i + 1)
		tasks = read_workfile(adapter, adir)
		assignments = OrderedDict(
			((assignment.uid, assignment.n), assignment) for assignment in tasks if isinstance(assignment, Assignment)
		).values()
		if not assignments:
			adapter.info("No work queued up.")
			continue
		cur_time_left = 0
		mersennes = True
		now = datetime.now()
		for assignment in assignments:
			time_left = work_estimate(adapter, adir, i if cpu_num is None else cpu_num, assignment)
			bits = max(32, int(assignment.sieve_depth))
			all_and_prp_cnt = False
			aprob = 0.0
			if assignment.work_type == PRIMENET.WORK_TYPE_FIRST_LL:
				work_type_str = "Lucas-Lehmer test"
				all_and_prp_cnt = True
				aprob += (
					(bits - 1)
					* 1.733
					* (1.04 if assignment.pminus1ed else 1.0)
					/ (log2(assignment.k) + log2(assignment.b) * assignment.n)
				)
			elif assignment.work_type == PRIMENET.WORK_TYPE_DBLCHK:
				work_type_str = "Double-check"
				all_and_prp_cnt = True
				aprob += (
					(bits - 1)
					* 1.733
					* ERROR_RATE
					* (1.04 if assignment.pminus1ed else 1.0)
					/ (log2(assignment.k) + log2(assignment.b) * assignment.n)
				)
			elif assignment.work_type == PRIMENET.WORK_TYPE_PRP:
				all_and_prp_cnt = True
				if not assignment.prp_dblchk:
					work_type_str = "PRP"
					aprob += (
						(bits - 1)
						* 1.733
						* (1.04 if assignment.pminus1ed else 1.0)
						/ (log2(assignment.k) + log2(assignment.b) * assignment.n)
					)
				else:
					work_type_str = "PRPDC"
					aprob += (
						(bits - 1)
						* 1.733
						* PRP_ERROR_RATE
						* (1.04 if assignment.pminus1ed else 1.0)
						/ (log2(assignment.k) + log2(assignment.b) * assignment.n)
					)
			elif assignment.work_type == PRIMENET.WORK_TYPE_FACTOR:
				work_type_str = "factor from 2^{0:.0f} to 2^{1:.0f}".format(assignment.sieve_depth, assignment.factor_to)
			elif assignment.work_type == PRIMENET.WORK_TYPE_PMINUS1:
				work_type_str = "P-1 B1={0}".format(assignment.B1)
			elif assignment.work_type == PRIMENET.WORK_TYPE_PFACTOR:
				work_type_str = "P-1"
			elif assignment.work_type == PRIMENET.WORK_TYPE_CERT:
				work_type_str = "Certify"
			prob += aprob
			if assignment.k != 1.0 or assignment.b != 2 or assignment.c != -1 or assignment.known_factors is not None:
				amersennes = mersennes = False
			else:
				amersennes = True
			buf = exponent_to_str(assignment) + (
				"/known_factors" if assignment.work_type == PRIMENET.WORK_TYPE_PRP and assignment.known_factors else ""
			)
			if time_left is None:
				adapter.info("%s, %s, Finish cannot be estimated", buf, work_type_str)
			else:
				cur_time_left += time_left
				time_left = timedelta(seconds=cur_time_left)
				adapter.info("%s, %s, %s (%s)", buf, work_type_str, time_left, (now + time_left).strftime("%c"))
			if all_and_prp_cnt:
				ll_and_prp_cnt += 1
				adapter.info(
					"The chance that the exponent (%s) you are testing will yield a %sprime is about 1 in %s (%s).",
					assignment.n,
					"Mersenne " if amersennes else "",
					format(int(1.0 / aprob), "n"),
					format(aprob, "%"),
				)
			digits(assignment)
	if ll_and_prp_cnt > 1:
		logging.info(
			"The chance that one of the %s exponents you are testing will yield a %sprime is about 1 in %s (%s).",
			ll_and_prp_cnt,
			"Mersenne " if mersennes else "",
			format(int(1.0 / prob), "n"),
			format(prob, "%"),
		)


def get_disk_usage(path):
	"""Calculate the total disk usage of all files in the given directory, excluding symbolic links."""
	total = 0
	for dirpath, _dirnames, filenames in os.walk(path):
		for filename in filenames:
			file = os.path.join(dirpath, filename)
			if not os.path.islink(file):
				total += os.path.getsize(file)
	return total


def check_disk_space(dirs):
	"""Check and log the disk space usage and availability, sending alerts if critical thresholds are reached."""
	usage = disk_usage(workdir)

	if options.worker_disk_space:
		worker_disk_space = options.worker_disk_space * 1024**3
		disk_space = min(usage.total, worker_disk_space * options.num_workers)
		usages = [get_disk_usage(adir) for adir in dirs]
		total = sum(usages)
		logging.debug("Disk space limit usage: %s", output_available(total, disk_space))
		precent = total / disk_space
		critical = (
			config.getfloat(SEC.PrimeNet, "disk_usage_critical") if config.has_option(SEC.PrimeNet, "disk_usage_critical") else 90
		)
		if precent * 100 >= critical:
			logging.warning(
				"Greater than %s%% or %s of the configured disk space limit used (%sB / %sB)",
				critical,
				format(precent, "%"),
				outputunit(total),
				outputunit(disk_space),
			)
			if not config.has_option(SEC.PrimeNet, "storage_usage_critical"):
				send_msg(
					"⚠️🗃️ {0:.1%} of the disk space limit used on {1}".format(precent, options.computer_id),
					"""{0:%} or {1}B of the configured {2}B ({3}B × {4}) disk space limit is used on your {5!r} computer.

Disk space usage:
{6}

Total limit usage: {7}
""".format(
						precent,
						outputunit(total),
						outputunit(disk_space),
						outputunit(worker_disk_space),
						options.num_workers,
						options.computer_id,
						"\n".join("Worker #{0}, {1!r}: {2}".format(i + 1, dirs[i], outputunit(u)) for i, u in enumerate(usages)),
						output_available(total, disk_space),
					),
					priority="2 (High)",
				)
				config.set(SEC.PrimeNet, "storage_usage_critical", str(True))
		else:
			config.remove_option(SEC.PrimeNet, "storage_usage_critical")

	logging.debug("Disk space available: %s", output_available(usage.free, usage.total))
	precent = usage.free / usage.total
	critical = (
		config.getfloat(SEC.PrimeNet, "disk_available_critical")
		if config.has_option(SEC.PrimeNet, "disk_available_critical")
		else 10
	)
	if precent * 100 <= critical:
		logging.warning(
			"Less than %s%% or only %s of the total disk space available (%sB / %sB)",
			critical,
			format(precent, "%"),
			outputunit(usage.free),
			outputunit(usage.total),
		)
		if not config.has_option(SEC.PrimeNet, "storage_available_critical"):
			send_msg(
				"🚨🗃️ Only {0:.1%} of the disk space available on {1}".format(precent, options.computer_id),
				"""Only {0:%} or {1}B of the total {2}B disk space is available on your {3!r} computer.

If the computer runs out of space, the program will likely be unable to generate the PRP proof files.

Disk space available: {4}
""".format(
					precent,
					outputunit(usage.free),
					outputunit(usage.total),
					options.computer_id,
					output_available(usage.free, usage.total),
				),
				priority="1 (Highest)",
			)
			config.set(SEC.PrimeNet, "storage_available_critical", str(True))
	else:
		config.remove_option(SEC.PrimeNet, "storage_available_critical")


def checksum_md5(filename):
	"""Calculate and return the MD5 checksum of a given file."""
	amd5 = md5()
	with open(filename, "rb") as f:
		for chunk in iter(lambda: f.read(256 * amd5.block_size), b""):
			amd5.update(chunk)
	return amd5.hexdigest()


PROOF_NUMBER_RE = re.compile(br"^(\()?([MF]?(\d+)|(?:(\d+)\*)?(\d+)\^(\d+)([+-]\d+))(?(1)\))(?:/(\d+(?:/\d+)*))?$")


def upload_proof(adapter, filename):
	"""Uploads a proof file to the server in chunks, resuming from the last uploaded position if interrupted."""
	max_chunk_size = config.getfloat(SEC.PrimeNet, "UploadChunkSize") if config.has_option(SEC.PrimeNet, "UploadChunkSize") else 5
	max_chunk_size = int(min(max(max_chunk_size, 1), 8) * 1024 * 1024)
	starttime = timeit.default_timer()
	try:
		with open(filename, "rb") as f:
			header = f.readline().rstrip(b"\n")
			if header != b"PRP PROOF":
				return False
			header, _, version = f.readline().rstrip(b"\n").partition(b"=")
			if header != b"VERSION" or int(version) not in {1, 2}:
				adapter.error("Error getting version number from proof header")
				return False
			header, _, hashlen = f.readline().rstrip(b"\n").partition(b"=")
			if header != b"HASHSIZE" or not 32 <= int(hashlen) <= 64:
				adapter.error("Error getting hash size from proof header")
				return False
			header, _, power = f.readline().rstrip(b"\n").partition(b"=")
			power, _, _power_mult = power.partition(b"x")
			if header != b"POWER" or not 0 < int(power) < 16:
				adapter.error("Error getting power from proof header")
				return False
			header = f.readline().rstrip(b"\n")
			if header.startswith(b"BASE="):
				header = f.readline().rstrip(b"\n")
			header, _, number = header.partition(b"=")
			if header != b"NUMBER":
				adapter.error("Error getting number from proof header")
				return False
			proof_number = PROOF_NUMBER_RE.match(number)
			if not proof_number:
				logging.error("Error parsing number: %r", number)
				return False
			_, number, exponent, _k, _b, _n, _c, _factors = proof_number.groups()
			if not exponent or not number.startswith(b"M"):
				logging.error("Only proof files for Mersenne numbers are supported")
				return False
			exponent = int(exponent)
			adapter.info("Proof file exponent is %s", exponent)
			filesize = os.path.getsize(filename)
			adapter.info(
				"Filesize of %r is %sB%s",
				filename,
				outputunit(filesize),
				" ({0}B)".format(outputunit(filesize, True)) if filesize >= 1000 else "",
			)
			fileHash = checksum_md5(filename)
			adapter.info("MD5 of %r is %s", filename, fileHash)

			while True:
				r = session.get(
					primenet_baseurl + "proof_upload/",
					params={"UserID": options.user_id, "Exponent": exponent, "FileSize": filesize, "FileMD5": fileHash},
					timeout=180,
				)
				result = r.json()
				if "error_status" in result:
					if result["error_status"] == 409:
						adapter.error("Proof %r already uploaded", filename)
						adapter.error("%s", result)
						return True
					adapter.error("Unexpected error during %r upload", filename)
					adapter.error("%s", result)
					return False
				r.raise_for_status()
				if "URLToUse" not in result:
					adapter.error("For proof %r, server response missing URLToUse:", filename)
					adapter.error("%s", result)
					return False
				if "need" not in result:
					adapter.error("For proof %r, server response missing need list:", filename)
					adapter.error("%s", result)
					return False

				origUrl = result["URLToUse"]
				baseUrl = "https" + origUrl[4:] if origUrl.startswith("http:") else origUrl
				pos, end = next((int(a), b) for a, b in result["need"].items())
				if pos > end or end >= filesize:
					adapter.error("For proof %r, need list entry bad:", filename)
					adapter.error("%s", result)
					return False

				if pos:
					adapter.info("Resuming from offset %s", pos)

				bytessent = 0
				while pos < end:
					f.seek(pos)
					size = min(end - pos + 1, max_chunk_size)
					chunk = f.read(size)
					bytessent += size
					response = session.post(
						baseUrl,
						params={"FileMD5": fileHash, "DataOffset": pos, "DataSize": len(chunk), "DataMD5": md5(chunk).hexdigest()},
						files={"Data": (None, chunk)},
						timeout=180,
					)
					result = response.json()
					if "error_status" in result:
						adapter.error("Unexpected error during %r upload", filename)
						adapter.error("%s", result)
						return False
					response.raise_for_status()
					if "FileUploaded" in result:
						adapter.info("Proof file %r successfully uploaded", filename)
						endtime = timeit.default_timer()
						totaltime = endtime - starttime
						adapter.info(
							"Uploaded %sB%s in %s, %sB/sec",
							outputunit(bytessent),
							" ({0}B)".format(outputunit(bytessent, True)) if bytessent >= 1000 else "",
							timedelta(seconds=totaltime),
							outputunit(bytessent / totaltime),
						)
						return True
					if "need" not in result:
						adapter.error("For proof %r, no entries in need list:", filename)
						adapter.error("%s", result)
						return False
					start, end = next((int(a), b) for a, b in result["need"].items())
					if start <= pos:
						adapter.error("For proof %r, sending data did not advance need list:", filename)
						adapter.error("%s", result)
						return False
					pos = start
					if pos > end or end >= filesize:
						adapter.error("For proof %r, need list entry bad:", filename)
						adapter.error("%s", result)
						return False
	except RequestException as e:
		logging.exception("%s", e, exc_info=options.debug)
		return False
	except (IOError, OSError) as e:
		logging.exception("Cannot open proof file %r: %s", filename, e, exc_info=options.debug)
		return False


def upload_proofs(adapter, adir, cpu_num):
	"""Uploads proof files from a given directory."""
	if config.has_option(SEC.PrimeNet, "ProofUploads") and not config.getboolean(SEC.PrimeNet, "ProofUploads"):
		return
	proof = os.path.join(adir, "proof")
	if not os.path.isdir(proof):
		adapter.debug("Proof directory %r does not exist", proof)
		return
	entries = glob.glob(os.path.join(proof, "*.proof"))
	if not entries:
		adapter.debug("No proof files to upload.")
		return
	if options.archive_dir:
		archive = os.path.join(adir, options.archive_dir)
		if not os.path.exists(archive):
			os.makedirs(archive)
	for entry in entries:
		filename = os.path.basename(entry)
		if upload_proof(adapter, entry):
			if options.archive_dir:
				shutil.move(entry, os.path.join(archive, filename))
			else:
				os.remove(entry)
		else:
			send_msg(
				"❌📜 Failed to upload the {0} proof file on {1}".format(filename, options.computer_id),
				"""Failed to upload the {0!r} PRP proof file on your {1!r} computer (worker #{2}).

Below is the last up to 10 lines of the {3!r} log file for AutoPrimeNet:

{4}

If you believe this is a bug with AutoPrimeNet, please create an issue: https://github.com/tdulcet/AutoPrimeNet/issues
""".format(entry, options.computer_id, cpu_num + 1, logfile, tail(logfile, 10)),
				priority="2 (High)",
			)


def aupload_proofs(dirs):
	"""Uploads proofs from the given directories."""
	for i, adir in enumerate(dirs):
		adapter = logging.LoggerAdapter(logger, {"cpu_num": i} if options.dirs else None)
		cpu_num = i if options.dirs else options.cpu
		upload_proofs(adapter, adir, cpu_num)


# TODO -- have people set their own program options for commented out portions
def program_options(send=False, start=-1, retry_count=0):
	"""Sets the program options on the PrimeNet server."""
	guid = get_guid(config)
	for tnum in range(start, options.num_workers):
		args = primenet_v5_bargs.copy()
		args["t"] = "po"
		args["g"] = guid
		# no value updates all cpu threads with given worktype
		args["c"] = "" if tnum < 0 else tnum
		if send:
			options_changed = False
			if len(set(work_preference)) == 1 if tnum < 0 else len(set(work_preference)) != 1:
				args["w"] = work_preference[max(0, tnum)]
				options_changed = True
			if tnum < 0:
				args["nw"] = options.num_workers
				# args["Priority"] = 1
				args["DaysOfWork"] = max(1, int(round(options.days_of_work)))
				args["DayMemory"] = options.day_night_memory
				args["NightMemory"] = options.day_night_memory
				# args["DayStartTime"] = 0
				# args["NightStartTime"] = 0
				# args["RunOnBattery"] = 1
				options_changed = True
		if not send or options_changed:
			retry = False
			logging.info("Exchanging program options with server")
			result = send_request(guid, args)
			if result is None:
				parser.error("Error while setting program options on mersenne.org")
			else:
				rc = int(result["pnErrorResult"])
				if rc == PRIMENET.ERROR_OK:
					pass
				else:
					if rc == PRIMENET.ERROR_UNREGISTERED_CPU:
						register_instance()
						retry = True
					elif rc == PRIMENET.ERROR_STALE_CPU_INFO:
						register_instance(guid)
						retry = True
					if not retry:
						parser.error("Error while setting program options on mersenne.org")
			if retry:
				if retry_count >= 2:
					logging.info("Retry count exceeded.")
					return None
				time.sleep(1 << retry_count)
				return program_options(send, tnum, retry_count + 1)
			if "w" in result:
				w = int(result["w"])
				awork_preference = int(options.work_preference[max(0, tnum)])
				aw = (
					next(key for key, value in CONVERT_DICT.items() if value == w)
					if awork_preference in CONVERT_DICT and w in CONVERT_DICT.values()
					else w
				)
				if awork_preference != aw:
					logging.warning("Work preference changed to %s", aw)
				if aw not in SUPPORTED:
					logging.critical("Unsupported work preference = %s for %s", aw, PROGRAM["name"])
					sys.exit(1)
				if tnum < 0:
					for i in range(options.num_workers):
						work_preference[i] = w
						options.work_preference[i] = str(aw)
						section = "Worker #{0}".format(i + 1) if options.num_workers > 1 else SEC.PrimeNet
						config.set(section, "WorkPreference", str(aw))
				else:
					work_preference[tnum] = w
					options.work_preference[tnum] = str(aw)
					section = "Worker #{0}".format(tnum + 1) if options.num_workers > 1 else SEC.PrimeNet
					config.set(section, "WorkPreference", str(aw))
			if "nw" in result:
				options.num_workers = int(result["nw"])
				config.set(SEC.PrimeNet, "NumWorkers", result["nw"])
			if "Priority" in result:
				config.set(SEC.PrimeNet, "Priority", result["Priority"])
			if "DaysOfWork" in result:
				options.days_of_work = float(result["DaysOfWork"])
				config.set(SEC.PrimeNet, "DaysOfWork", result["DaysOfWork"])
			if "DayMemory" in result and "NightMemory" in result:
				memory = max(int(result[x]) for x in ("DayMemory", "NightMemory"))
				options.day_night_memory = memory
				config.set(SEC.PrimeNet, "Memory", str(memory))
			if "RunOnBattery" in result:
				config.set(SEC.PrimeNet, "RunOnBattery", result["RunOnBattery"])
			if send:
				config.set(
					SEC.Internals,
					"SrvrP00",
					str(config.getint(SEC.Internals, "SrvrP00") + 1 if config.has_option(SEC.Internals, "SrvrP00") else 0),
				)
			else:
				config.set(SEC.Internals, "SrvrP00", result["od"])
	return None


def register_instance(guid=None):
	"""Register the computer with the PrimeNet server."""
	# register the instance to server, guid is the instance identifier
	hardware_id = md5((options.cpu_brand + str(uuid.getnode())).encode("utf-8")).hexdigest()  # similar as MPrime
	if config.has_option(SEC.PrimeNet, "HardwareGUID"):
		hardware_id = config.get(SEC.PrimeNet, "HardwareGUID")
	else:
		config.set(SEC.PrimeNet, "HardwareGUID", hardware_id)
	args = primenet_v5_bargs.copy()
	args["t"] = "uc"  # update compute command
	if guid is None:
		guid = create_new_guid()
	args["g"] = guid
	args["hg"] = hardware_id  # 32 hex char (128 bits)
	args["wg"] = ""  # only filled on Windows by MPrime
	args["a"] = generate_application_str()
	if config.has_option(SEC.PrimeNet, "sw_version"):
		args["a"] = config.get(SEC.PrimeNet, "sw_version")
	args["c"] = options.cpu_brand  # CPU model (len between 8 and 64)
	args["f"] = options.cpu_features  # CPU option (like asimd, max len 64)
	args["L1"] = options.cpu_l1_cache_size  # L1 cache size in KBytes
	args["L2"] = options.cpu_l2_cache_size  # L2 cache size in KBytes
	# if smaller or equal to 256,
	# server refuses to give LL assignment
	args["np"] = options.num_cores  # number of cores
	args["hp"] = options.cpu_hyperthreads  # number of hyperthreading cores
	args["m"] = options.memory  # number of megabytes of physical memory
	args["s"] = options.cpu_speed  # CPU frequency
	args["h"] = options.cpu_hours
	args["r"] = config.getint(SEC.Internals, "RollingAverage") if config.has_option(SEC.Internals, "RollingAverage") else 1000
	if options.cpu_l3_cache_size:
		args["L3"] = options.cpu_l3_cache_size
	if options.user_id:
		args["u"] = options.user_id
	if options.computer_id:
		args["cn"] = options.computer_id  # truncate to 20 char max
	logging.info("Updating computer information on the server")
	result = send_request(guid, args)
	if result is None:
		parser.error("Error while registering on mersenne.org")
	else:
		rc = int(result["pnErrorResult"])
		if rc == PRIMENET.ERROR_OK:
			pass
		else:
			parser.error("Error while registering on mersenne.org")
	# Save program options in case they are changed by the PrimeNet server.
	options.user_id = result["u"]
	config.set(SEC.PrimeNet, "username", options.user_id)
	options.computer_id = result["cn"]
	config.set(SEC.PrimeNet, "ComputerID", options.computer_id)
	config.set(SEC.PrimeNet, "user_name", result["un"])
	options_counter = int(result["od"])
	guid = result["g"]
	config_write(config, guid)
	# if options_counter == 1:
	# program_options()
	program_options(True)
	if options_counter > config.getint(SEC.Internals, "SrvrP00"):
		program_options()
	config_write(config)
	logging.info(
		"""GUID %s correctly registered with the following features:
Username: %s
Computer name: %s
CPU/GPU model: %s
CPU features: %s
CPU L1 Cache size: %s KiB
CPU L2 Cache size: %s KiB
CPU cores: %s
CPU threads per core: %s
CPU/GPU frequency/speed: %s MHz
Total RAM: %s MiB
To change these values, please rerun the program with different options
You can see the result in this page:
https://www.mersenne.org/editcpu/?g=%s""",
		guid,
		options.user_id,
		options.computer_id,
		options.cpu_brand,
		options.cpu_features,
		options.cpu_l1_cache_size,
		options.cpu_l2_cache_size,
		options.num_cores,
		options.cpu_hyperthreads,
		options.cpu_speed,
		options.memory,
		guid,
	)


def assignment_unreserve(adapter, assignment, retry_count=0):
	"""Unreserves an assignment from the PrimeNet server."""
	guid = get_guid(config)
	if guid is None:
		adapter.error("Cannot unreserve, the registration is not done")
		return False
	if not assignment or not assignment.uid:
		return True
	args = primenet_v5_bargs.copy()
	args["t"] = "au"
	args["g"] = guid
	args["k"] = assignment.uid
	retry = False
	adapter.info("Unreserving %s", exponent_to_str(assignment))
	result = send_request(guid, args)
	if result is None:
		retry = True
	else:
		rc = int(result["pnErrorResult"])
		if rc == PRIMENET.ERROR_OK:
			return True
		if rc == PRIMENET.ERROR_INVALID_ASSIGNMENT_KEY:
			return True
		if rc == PRIMENET.ERROR_UNREGISTERED_CPU:
			register_instance()
			retry = True
	if retry:
		if retry_count >= 2:
			adapter.info("Retry count exceeded.")
			return False
		time.sleep(1 << retry_count)
		return assignment_unreserve(adapter, assignment, retry_count + 1)
	return False


def unreserve(dirs, p):
	"""Unreserve a specific exponent from the workfile."""
	adapter = logging.LoggerAdapter(logger, None)
	for adir in dirs:
		workfile = os.path.join(adir, options.worktodo_file)
		with LockFile(workfile):
			tasks = list(read_workfile(adapter, adir))
			found = changed = False
			for assignment in tasks:
				if isinstance(assignment, Assignment) and assignment.n == p:
					if assignment_unreserve(adapter, assignment):
						tasks = (
							task
							for task in tasks
							if not isinstance(task, Assignment)
							or (task.uid != assignment.uid if assignment.uid else task.n != assignment.n)
						)
						changed = True
					found = True
					break
			if found:
				if changed:
					write_workfile(adir, tasks)
				break
	else:
		logging.error("Error unreserving exponent: %s not found in workfile%s", p, "s" if len(dirs) != 1 else "")


def get_proof_data(adapter, assignment_aid, file):
	"""Downloads proof data for a given assignment and writes it to a file."""
	max_chunk_size = (
		int(config.getfloat(SEC.PrimeNet, "DownloadChunkSize") * 1024 * 1024)
		if config.has_option(SEC.PrimeNet, "DownloadChunkSize")
		else None
	)
	starttime = timeit.default_timer()
	try:
		r = session.get(primenet_baseurl + "proof_get_data/", params={"aid": assignment_aid}, timeout=180, stream=True)
		r.raise_for_status()
		length = int(r.headers["Content-Length"])
		if hasattr(os, "posix_fallocate"):  # Python 3.3+, Linux
			os.posix_fallocate(file.fileno(), 0, length - 32)
		amd5 = next(r.iter_content(chunk_size=32))
		for chunk in r.iter_content(chunk_size=max_chunk_size):
			if chunk:
				file.write(chunk)
	except RequestException as e:
		logging.exception("%s", e, exc_info=options.debug)
		return None
	endtime = timeit.default_timer()
	totaltime = endtime - starttime
	adapter.info(
		"Downloaded %sB%s in %s, %sB/sec",
		outputunit(length),
		" ({0}B)".format(outputunit(length, True)) if length >= 1000 else "",
		timedelta(seconds=totaltime),
		outputunit(length / totaltime),
	)
	return amd5


IS_HEX_RE = re.compile(br"^[0-9a-fA-F]*$")  # string.hexdigits


def download_cert(adapter, adir, filename, assignment):
	"""Downloads and verifies the certification starting value for a given assignment."""
	adapter.info("Downloading CERT starting value for %s to %r", exponent_to_str(assignment), filename)
	with tempfile.NamedTemporaryFile("wb", dir=adir, delete=False) as f:
		amd5 = get_proof_data(adapter, assignment.uid, f)
	if not amd5 or not IS_HEX_RE.match(amd5):
		adapter.error("Error getting CERT starting value")
		os.remove(f.name)
		return False
	amd5 = amd5.decode("utf-8").upper()
	residue_md5 = checksum_md5(f.name).upper()
	if amd5 != residue_md5:
		adapter.error("MD5 of downloaded starting value %s does not match %s", residue_md5, amd5)
		os.remove(f.name)
		return False
	os.rename(f.name, filename)
	adapter.info("CERT starting value %r successfully downloaded", filename)
	return True


def download_certs(adapter, adir, tasks):
	"""Downloads certification files for given assignments if they do not already exist."""
	for assignment in tasks:
		if isinstance(assignment, Assignment) and assignment.work_type == PRIMENET.WORK_TYPE_CERT:
			filename = os.path.join(adir, "{0}.cert".format(exponent_to_str(assignment)))
			if not os.path.exists(filename):
				if not download_cert(adapter, adir, filename, assignment):
					echk = (
						config.getint(SEC.Internals, "CertErrorCount") if config.has_option(SEC.Internals, "CertErrorCount") else 0
					) + 1
					config.set(SEC.Internals, "CertErrorCount", str(echk))
					if echk < 8:
						adapter.info("Will retry downloading CERT later")
					else:
						adapter.info("Abandoning CERT of M%s", assignment.n)
						unreserve([adir], assignment.n)
						config.remove_option(SEC.Internals, "CertErrorCount")


def get_assignment(
	adapter, cpu_num, assignment_num=None, get_cert_work=None, min_exp=None, max_exp=None, recover_all=False, retry_count=0
):
	"""Get a new assignment from the PrimeNet server."""
	guid = get_guid(config)
	args = primenet_v5_bargs.copy()
	args["t"] = "ga"  # transaction type
	args["g"] = guid
	args["c"] = cpu_num
	args["a"] = assignment_num
	if assignment_num is None:
		if get_cert_work:
			args["cert"] = get_cert_work
		if options.worker_disk_space:
			args["disk"] = "{0:f}".format(options.worker_disk_space)
		if min_exp:
			args["min"] = min_exp
		if max_exp:
			args["max"] = max_exp
		if options.min_bit:
			args["sf"] = options.min_bit
		if options.max_bit:
			args["ef"] = options.max_bit
	elif recover_all:
		args["all"] = 1
	# adapter.debug("Fetching using v5 API")
	supported = frozenset((
		PRIMENET.WORK_TYPE_FACTOR,
		PRIMENET.WORK_TYPE_PFACTOR,
		PRIMENET.WORK_TYPE_FIRST_LL,
		PRIMENET.WORK_TYPE_DBLCHK,
		PRIMENET.WORK_TYPE_PRP,
		PRIMENET.WORK_TYPE_CERT,
	))
	retry = False
	if (assignment_num is None or assignment_num) and not get_cert_work:
		adapter.info("Getting assignment from server")
	r = send_request(guid, args)
	if r is None:
		retry = True
	else:
		rc = int(r["pnErrorResult"])
		if rc == PRIMENET.ERROR_OK:
			pass
		else:
			if rc == PRIMENET.ERROR_UNREGISTERED_CPU:
				register_instance()
				retry = True
			elif rc == PRIMENET.ERROR_STALE_CPU_INFO:
				register_instance(guid)
				retry = True
			elif rc == PRIMENET.ERROR_CPU_CONFIGURATION_MISMATCH:
				register_instance(guid)
				retry = True
			if not retry:
				return None
	if retry:
		if retry_count >= 2:
			adapter.info("Retry count exceeded.")
			return None
		time.sleep(1 << retry_count)
		return get_assignment(adapter, cpu_num, assignment_num, get_cert_work, min_exp, max_exp, recover_all, retry_count + 1)
	if assignment_num is not None and not assignment_num:
		return int(r["a"])
	assignment = Assignment(int(r["w"]))
	assignment.uid = r["k"]
	assignment.n = int(r["n"])
	if assignment.n < 15000000 and assignment.work_type in {
		PRIMENET.WORK_TYPE_FACTOR,
		PRIMENET.WORK_TYPE_PFACTOR,
		PRIMENET.WORK_TYPE_FIRST_LL,
		PRIMENET.WORK_TYPE_DBLCHK,
	}:
		adapter.error("Server sent bad exponent: %s.", assignment.n)
		return None
	if assignment.work_type not in supported:
		adapter.error("Returned assignment from server is not a supported worktype %s.", assignment.work_type)
		# TODO: Unreserve assignment
		# assignment_unreserve()
		# return None
	if assignment.work_type == PRIMENET.WORK_TYPE_FIRST_LL:
		assignment.sieve_depth = float(r["sf"])
		assignment.pminus1ed = int(r["p1"])
		if options.gpuowl:  # GpuOwl
			adapter.warning("First time LL tests are not supported with the latest versions of GpuOwl")
	elif assignment.work_type == PRIMENET.WORK_TYPE_DBLCHK:
		assignment.sieve_depth = float(r["sf"])
		assignment.pminus1ed = int(r["p1"])
		if options.gpuowl:  # GpuOwl
			adapter.warning("Double check LL tests are not supported with the latest versions of GpuOwl")
	elif assignment.work_type == PRIMENET.WORK_TYPE_PRP:
		assignment.prp_dblchk = "dc" in r
		assignment.k = float(r["A"])
		assignment.b = int(r["b"])
		assignment.c = int(r["c"])
		if "sf" in r and "saved" in r:
			assignment.sieve_depth = float(r["sf"])
			assignment.tests_saved = float(r["saved"])
			if "base" in r and "rt" in r:
				assignment.prp_base = int(r["base"])
				assignment.prp_residue_type = int(r["rt"])
				# Mlucas
				if not (options.cudalucas or options.gpuowl):
					if assignment.prp_base != 3:
						adapter.error("PRP base (%s) is not 3", assignment.prp_base)
					if assignment.prp_residue_type not in {1, 5}:
						adapter.error("PRP residue type (%s) is not 1 or 5", assignment.prp_residue_type)
					# TODO: Unreserve assignment
					# assignment_unreserve()
		if "kf" in r:
			# Workaround Mlucas bug: https://github.com/primesearch/Mlucas/issues/30
			if not (options.cudalucas or options.gpuowl):
				if not assignment.prp_base:
					assignment.prp_base = 3
				if not assignment.prp_residue_type:
					assignment.prp_residue_type = 5
			assignment.known_factors = tuple(map(int, r["kf"].split(",")))
		if "pp" in r:
			config.set(SEC.PrimeNet, "ProofPower", r["pp"])
		else:
			config.remove_option(SEC.PrimeNet, "ProofPower")
		if "ppm" in r:
			config.set(SEC.PrimeNet, "ProofPowerMult", r["ppm"])
		else:
			config.remove_option(SEC.PrimeNet, "ProofPowerMult")
		if "ph" in r:
			config.set(SEC.PrimeNet, "ProofHashLength", r["ph"])
		else:
			config.remove_option(SEC.PrimeNet, "ProofHashLength")
	elif assignment.work_type == PRIMENET.WORK_TYPE_FACTOR:
		assignment.sieve_depth = float(r["sf"])
		assignment.factor_to = float(r["ef"])
	elif assignment.work_type == PRIMENET.WORK_TYPE_PFACTOR:
		assignment.k = float(r["A"])
		assignment.b = int(r["b"])
		assignment.c = int(r["c"])
		assignment.sieve_depth = float(r["sf"])
		assignment.tests_saved = float(r["saved"])
	elif assignment.work_type == PRIMENET.WORK_TYPE_PMINUS1:
		assignment.k = float(r["A"])
		assignment.b = int(r["b"])
		assignment.c = int(r["c"])
		assignment.B1 = int(r["B1"])
		assignment.B2 = int(r["B2"])
	elif assignment.work_type == PRIMENET.WORK_TYPE_CERT:
		assignment.k = float(r["A"])
		assignment.b = int(r["b"])
		assignment.c = int(r["c"])
		assignment.cert_squarings = int(r["ns"])
	else:
		adapter.critical("Received unknown worktype: %s.", assignment.work_type)
		sys.exit(1)
	adapter.info("Got assignment %s: %s", assignment.uid, exponent_to_text(assignment))
	return assignment


def get_cert_work(adapter, adir, cpu_num, current_time, progress, tasks):
	"""Manages the retrieval and assignment of certification work based on configuration and resource limits."""
	if config.has_option(SEC.PrimeNet, "QuitGIMPS") and config.getboolean(SEC.PrimeNet, "QuitGIMPS"):
		return
	if not options.cert_work or options.days_of_work <= 0 or options.cpu_hours <= 12:
		return
	max_cert_assignments = 3 if options.cert_cpu_limit >= 50 else 1
	cert_assignments = sum(
		1 for assignment in tasks if isinstance(assignment, Assignment) and assignment.work_type == PRIMENET.WORK_TYPE_CERT
	)
	workfile = os.path.join(adir, options.worktodo_file)
	if cert_assignments >= max_cert_assignments:
		adapter.debug(
			"%s ≥ %s CERT assignments already in %r, not getting new work", cert_assignments, max_cert_assignments, workfile
		)
		return

	section = "Worker #{0}".format(cpu_num + 1) if options.num_workers > 1 else SEC.Internals
	cpu_limit_remaining = (
		config.getfloat(section, "CertDailyCPURemaining")
		if config.has_option(section, "CertDailyCPURemaining")
		else options.cert_cpu_limit
	)
	last_update = (
		config.getint(SEC.Internals, "CertDailyRemainingLastUpdate")
		if config.has_option(SEC.Internals, "CertDailyRemainingLastUpdate")
		else 0
	)
	days = max(0, (current_time - last_update) / (24 * 60 * 60))
	cpu_limit_remaining += days * options.cert_cpu_limit
	cpu_limit_remaining = min(cpu_limit_remaining, options.cert_cpu_limit)
	config.set(section, "CertDailyCPURemaining", str(cpu_limit_remaining))
	if cpu_limit_remaining <= 0:
		adapter.debug("CERT daily work limit already used, not getting new work")
		return

	assignment = next((assignment for assignment in tasks if isinstance(assignment, Assignment)), None)
	percent = None
	if progress is not None:
		percent, _time_left, _, _ = progress
	if options.cert_cpu_limit < 50 and (
		(assignment is not None and assignment.work_type in {PRIMENET.WORK_TYPE_PMINUS1, PRIMENET.WORK_TYPE_PFACTOR})
		or (percent is not None and percent > 0.85)
	):
		return

	if config.has_option(SEC.PrimeNet, "CertWorker") and config.getint(SEC.PrimeNet, "CertWorker") != cpu_num + 1:
		return
	min_exp = config.getint(SEC.PrimeNet, "CertMinExponent") if config.has_option(SEC.PrimeNet, "CertMinExponent") else 50000000
	max_exp = config.getint(SEC.PrimeNet, "CertMaxExponent") if config.has_option(SEC.PrimeNet, "CertMaxExponent") else None
	cert_quantity = config.getint(SEC.PrimeNet, "CertQuantity") if config.has_option(SEC.PrimeNet, "CertQuantity") else 1
	changed = False  # with io.open(workfile, "a", encoding="utf-8") as file:
	for num_certs in range(1, 5 + 1):
		test = get_assignment(adapter, cpu_num, get_cert_work=max(1, options.cert_cpu_limit), min_exp=min_exp, max_exp=max_exp)
		if test is None:
			break
		if test.work_type != PRIMENET.WORK_TYPE_CERT:
			adapter.error("Received unknown work type (expected 200): %s", test.work_type)
			break
		tasks.appendleft(test)  # append
		new_task = output_assignment(test)
		adapter.debug("New assignment: %r", new_task)
		changed = True  # file.write(new_task + "\n")

		# TODO: Something better here
		cpu_quota_used = test.cert_squarings / 110000
		cpu_quota_used *= 2.1 ** log2(test.n / 97300000)
		cpu_limit_remaining -= cpu_quota_used
		config.set(section, "CertDailyCPURemaining", str(cpu_limit_remaining))

		if cpu_limit_remaining <= 0:
			break
		if test.n < 50000000:
			continue
		if num_certs < cert_quantity:
			continue
		if options.cert_cpu_limit < 50:
			break
	if changed:
		write_workfile(adir, tasks)


LUCAS_RE = re.compile(
	r"^M\( ([0-9]{6,}) \)(P|C, (0x[0-9a-f]{16})), offset = ([0-9]+), n = ([0-9]{3,})K, (CUDALucas v[^\s,]+)(?:, AID: ([0-9A-F]{32}))?$"
)
PM1_RE = re.compile(
	r"^M([0-9]{6,}) (?:has a factor: ([0-9]+)|found no factor) \(P-1, B1=([0-9]+), B2=([0-9]+), e=([0-9]+), n=([0-9]{3,})K(?:, aid=([0-9A-F]{32}))? (CUDAPm1 v[^\s)]+)\)$"
)


def cuda_result_to_json(resultsfile, sendline):
	"""Converts CUDALucas and CUDAPm1 results to JSON format."""
	# CUDALucas and CUDAPm1

	# sendline example: 'M( 108928711 )C, 0x810d83b6917d846c, offset = 106008371, n = 6272K, CUDALucas v2.06, AID: 02E4F2B14BB23E2E4B95FC138FC715A8'
	# sendline example: 'M( 108928711 )P, offset = 106008371, n = 6272K, CUDALucas v2.06, AID: 02E4F2B14BB23E2E4B95FC138FC715A8'
	ar = {}
	lucas_res = LUCAS_RE.match(sendline)
	pm1_res = PM1_RE.match(sendline)

	if lucas_res:
		exponent, status, res64, shift_count, fft_length, aprogram, aid = lucas_res.groups()
		ar["status"] = status[0]
		ar["worktype"] = "LL"  # CUDALucas only does LL tests
		if status.startswith("C"):
			ar["res64"] = res64[2:].upper()
		ar["shift-count"] = int(shift_count)
	elif pm1_res:
		exponent, factor, b1, b2, brent_suyama, fft_length, aid, aprogram = pm1_res.groups()
		ar["status"] = "F" if factor else "NF"
		ar["worktype"] = "P-1"
		if factor:
			ar["factors"] = [factor]
		b1 = int(b1)
		ar["b1"] = b1
		b2 = int(b2)
		if b2 > b1:
			ar["b2"] = b2
			brent_suyama = int(brent_suyama)
			if brent_suyama > 2:
				ar["brent-suyama"] = brent_suyama
	else:
		logging.error("Unable to parse entry in %r: %s", resultsfile, sendline)
		return None

	ar["exponent"] = int(exponent)
	ar["fft-length"] = int(fft_length) << 10
	ar["program"] = program = {}
	program["name"], program["version"] = aprogram.split(None, 1)
	if aid:
		ar["aid"] = aid
	ar["result"] = sendline
	return ar


GHZDAYS_RE = re.compile(r"CPU credit is ([0-9]+(?:\.[0-9]+)?) GHz-days")


def report_result(adapter, ar, message, assignment, result_type, tasks, retry_count=0):
	"""Submit one result line using v5 API, will be attributed to the computer identified by guid."""
	"""Return False if the submission should be retried"""
	guid = get_guid(config)
	if guid is None:
		adapter.error("Cannot submit results, the registration is not done")
		return None
	# JSON is required because assignment_id is necessary in that case
	# and it is not present in old output format.
	# adapter.debug("Submitting using v5 API")

	args = primenet_v5_bargs.copy()
	args["t"] = "ar"  # assignment result
	args["g"] = guid
	args["k"] = assignment.uid  # assignment id
	# message is the complete JSON string
	args["m"] = message
	args["r"] = result_type  # result type
	args["n"] = assignment.n
	if result_type in {PRIMENET.AR_LL_RESULT, PRIMENET.AR_LL_PRIME}:
		args["d"] = 1
		if result_type == PRIMENET.AR_LL_RESULT:
			args["rd"] = ar["res64"].strip().zfill(16)
		args["sc"] = ar["shift-count"]
		args["ec"] = ar.get("error-code", "0" * 8)
	elif result_type in {PRIMENET.AR_PRP_RESULT, PRIMENET.AR_PRP_PRIME}:
		args["d"] = 1
		args.update((("A", "{0:.0f}".format(assignment.k)), ("b", assignment.b), ("c", assignment.c)))
		if result_type == PRIMENET.AR_PRP_RESULT:
			args["rd"] = ar["res64"].strip().zfill(16)
			args["rt"] = ar["residue-type"]
		args["ec"] = ar.get("error-code", "0" * 8)
		if "known-factors" in ar:
			args["nkf"] = len(ar["known-factors"])
		args["base"] = ar["worktype"][4:]  # worktype == PRP-base
		if "shift-count" in ar:
			args["sc"] = ar["shift-count"]
		# 1 if Gerbicz error checking used in PRP test
		args["gbz"] = 1
		if "proof" in ar:
			proof = ar["proof"]
			if proof["power"]:
				args["pp"] = proof["power"]
				args["ph"] = proof["md5"]
	elif result_type in {PRIMENET.AR_TF_FACTOR, PRIMENET.AR_TF_NOFACTOR}:
		args["d"] = (
			1
			if not any(
				task.k == assignment.k and task.b == assignment.b and task.n == assignment.n and task.c == assignment.c
				for task in tasks
				if isinstance(task, Assignment)
			)
			else 0
		)
		args["sf"] = ar["bitlo"]
		if ar["rangecomplete"]:
			args["ef"] = ar["bithi"]
		if result_type == PRIMENET.AR_TF_FACTOR:
			args["f"] = ",".join(ar["factors"])
	elif result_type in {PRIMENET.AR_P1_FACTOR, PRIMENET.AR_P1_NOFACTOR}:
		args["d"] = (
			1
			if result_type == PRIMENET.AR_P1_FACTOR
			or not any(
				task.k == assignment.k and task.b == assignment.b and task.n == assignment.n and task.c == assignment.c
				for task in tasks
				if isinstance(task, Assignment)
			)
			else 0
		)
		args.update((("A", "{0:.0f}".format(assignment.k)), ("b", assignment.b), ("c", assignment.c)))
		args["B1"] = ar["B1"] if "B1" in ar else ar["b1"]
		if "b2" in ar or "B2" in ar:
			args["B2"] = ar["B2"] if "B2" in ar else ar["b2"]
		if result_type == PRIMENET.AR_P1_FACTOR:
			args["f"] = ",".join(ar["factors"])
	elif result_type == PRIMENET.AR_CERT:
		args["d"] = 1
		args.update((("A", "{0:.0f}".format(assignment.k)), ("b", assignment.b), ("c", assignment.c)))
		args["s3"] = ar["sha3-hash"]
		args["ec"] = ar.get("error-code", "0" * 8)
		if "shift-count" in ar:
			args["sc"] = ar["shift-count"]
	if "fft-length" in ar:
		args["fftlen"] = ar["fft-length"]

	result = send_request(guid, args)
	if result is None:
		pass
		# if this happens, the submission can be retried
		# since no answer has been received from the server
		# return False
	else:
		rc = int(result["pnErrorResult"])
		ghd = 0
		if rc == PRIMENET.ERROR_OK:
			ghzdays = GHZDAYS_RE.search(result["pnErrorDetail"])
			if ghzdays:
				ghd = float(ghzdays.group(1))
			else:
				adapter.warning("Unable to find GHz-days credit value in response")
			adapter.debug("Result correctly send to server")
			return rc, ghd
		if rc == PRIMENET.ERROR_UNREGISTERED_CPU:
			# should register again and retry
			register_instance()
			# return False
		elif rc == PRIMENET.ERROR_STALE_CPU_INFO:
			register_instance(guid)
		# In all other error case, the submission must not be retried
		elif rc == PRIMENET.ERROR_INVALID_ASSIGNMENT_KEY:
			# TODO: Delete assignment from workfile if it is not done
			return rc, ghd
		elif rc == PRIMENET.ERROR_WORK_NO_LONGER_NEEDED:
			# TODO: Delete assignment from workfile if it is not done
			return rc, ghd
		elif rc == PRIMENET.ERROR_NO_ASSIGNMENT:
			# TODO: Delete assignment from workfile if it is not done
			return rc, ghd
		elif rc == PRIMENET.ERROR_INVALID_RESULT_TYPE:
			return rc, ghd
		elif rc == PRIMENET.ERROR_INVALID_PARAMETER:
			adapter.error(
				"INVALID PARAMETER: This may be a bug in the program, please create an issue: https://github.com/tdulcet/AutoPrimeNet/issues"
			)
			return None

	if retry_count >= 2:
		adapter.info("Retry count exceeded.")
		return None
	time.sleep(1 << retry_count)
	return report_result(adapter, ar, message, assignment, result_type, tasks, retry_count + 1)


def submit_mersenne_ca_results(adapter, lines, retry_count=0):
	"""Submit results for exponents over 1,000,000,000 using https://www.mersenne.ca/submit-results.php."""
	length = len(lines)
	adapter.info("Submitting %s results to mersenne.ca", length)
	retry = rejected = False
	try:
		r = session.post(
			mersenne_ca_baseurl + "submit-results.php",
			params={"json": 1},
			data={"gimps_login": options.user_id},
			files={"results_file": ("results.json.txt", "\n".join(lines))},
			timeout=180,
		)
		r.raise_for_status()
		result = r.json()
	except RequestException as e:
		logging.exception("%s", e, exc_info=options.debug)
		retry = True
	else:
		adapter.info("mersenne.ca response:")
		adapter.info(result["user_message"])
		results = result["results"]
		if results["unknown"]:
			adapter.error("Unknown %s result%s.", results["unknown"], "s" if results["unknown"] != 1 else "")
		if results["rejected"]:
			rejected = result["lines"]["rejected"]
			adapter.error(
				"Rejected %s result%s: {2}",
				results["rejected"],
				"s" if results["rejected"] != 1 else "",
				", ".join(
					"{0} ({1:n} result{2})".format(reason, len(lines), "s" if len(lines) != 1 else "")
					for reason, lines in rejected.items()
				),
			)
		accepted = sum(results["accepted"].values())
		adapter.info("Submitted %s result%s to mersenne.ca.", accepted, "s" if accepted != 1 else "")
		factors = results["factors"]["new"]
		adapter.info(
			"Total credit is %s GHz-days%s.",
			format(sum(results["ghd"].values()), "n"),
			", found {0:n} new factor{1}".format(factors, "s" if factors != 1 else "") if factors else "",
		)
	if retry:
		if retry_count >= 2:
			logging.info("Retry count exceeded.")
			return None
		time.sleep(1 << retry_count)
		return submit_mersenne_ca_results(adapter, lines, retry_count + 1)
	return rejected


SCRIPT = {
	"name": "AutoPrimeNet",  # os.path.basename(sys.argv[0])
	"version": VERSION,
	"interpreter": {
		"interpreter": "Python",
		"implementation": platform.python_implementation(),
		"version": platform.python_version(),
	},
	"os": get_os(),
}
CUDA_RESULTPATTERN = re.compile(r"CUDALucas v|CUDAPm1 v")


def parse_result(adapter, adir, resultsfile, sendline):
	"""Parses the result from a given sendline, processes it, and sends the appropriate response to the server."""
	if CUDA_RESULTPATTERN.search(sendline):  # CUDALucas or CUDAPm1
		ar = cuda_result_to_json(resultsfile, sendline)
	else:  # Mlucas or GpuOwl
		try:
			ar = json.loads(sendline)
		except JSONDecodeError as e:
			adapter.exception("Unable to decode entry in %r: %s", resultsfile, e, exc_info=options.debug)
			# Mlucas
			if "Program: E" in sendline:
				adapter.info("Please upgrade to Mlucas v19 or greater.")
			return None

	program = ar["program"]
	aprogram = "{0} {1}".format(program["name"], program["version"])
	# adapter.debug("Program: %s", aprogram)
	config.set(SEC.Internals, "program", aprogram)

	user = ar.setdefault("user", options.user_id)
	computer = ar.setdefault("computer", options.computer_id)
	ar["script"] = SCRIPT
	message = json.dumps(ar, ensure_ascii=False, separators=(",", ":"))

	assignment = Assignment()
	assignment.uid = ar.get("aid", 0)
	if "k" in ar and "b" in ar and "n" in ar and "c" in ar:
		assignment.k = ar["k"]
		assignment.b = ar["b"]
		assignment.n = ar["n"]
		assignment.c = ar["c"]
	elif "exponent" in ar:
		assignment.n = int(ar["exponent"])
	if "known-factors" in ar:
		assignment.known_factors = tuple(map(int, ar["known-factors"]))

	worktype = ar["worktype"]
	if worktype == "LL":
		result_type = PRIMENET.AR_LL_PRIME if ar["status"] == "P" else PRIMENET.AR_LL_RESULT
		# ar["status"] == "C"
	elif worktype.startswith("PRP"):
		result_type = PRIMENET.AR_PRP_PRIME if ar["status"] == "P" else PRIMENET.AR_PRP_RESULT
		# ar["status"] == "C"
	elif worktype in {"P-1", "PM1"}:
		result_type = PRIMENET.AR_P1_FACTOR if ar["status"] == "F" else PRIMENET.AR_P1_NOFACTOR
		# ar["status"] == "NF"
	elif worktype == "TF":
		result_type = PRIMENET.AR_TF_FACTOR if ar["status"] == "F" else PRIMENET.AR_TF_NOFACTOR
		# ar["status"] == "NF"
	elif worktype in {"Cert", "CERT"}:
		result_type = PRIMENET.AR_CERT
	else:
		adapter.error("Unsupported worktype %s", worktype)
		return None

	buf = "" if not user else "UID: {0}, ".format(user) if not computer else "UID: {0}/{1}, ".format(user, computer)
	if result_type in {PRIMENET.AR_LL_RESULT, PRIMENET.AR_LL_PRIME}:
		if result_type == PRIMENET.AR_LL_RESULT:
			buf += "M{0} is not prime. Res64: {1}. {2},{3},{4}".format(
				assignment.n, ar["res64"], ar.get("security-code", "-"), ar["shift-count"], ar.get("error-code", "0" * 8)
			)
		else:
			buf += "M{0} is prime! {1},{2}".format(assignment.n, ar.get("security-code", "-"), ar.get("error-code", "0" * 8))
	elif result_type in {PRIMENET.AR_PRP_RESULT, PRIMENET.AR_PRP_PRIME}:
		prp_base = int(worktype[4:])
		if result_type == PRIMENET.AR_PRP_RESULT:
			residue_type = ar["residue-type"]
			buf += "{0} is not prime.  {1}{2}RES64: {3}.".format(
				assignment_to_str(assignment),
				"Base-{0} ".format(prp_base) if prp_base != 3 else "",
				"Type-{0} ".format(residue_type) if residue_type != 1 else "",
				ar["res64"],
			)
		else:
			buf += "{0} is a probable prime{1}!".format(
				assignment_to_str(assignment), " ({0}-PRP)".format(prp_base) if prp_base != 3 else ""
			)
		buf += " {0},{1}{2}".format(
			ar.get("security-code", "-"),
			"{0},".format(ar["shift-count"]) if "shift-count" in ar else "",
			ar.get("error-code", "0" * 8),
		)
	elif result_type in {PRIMENET.AR_TF_FACTOR, PRIMENET.AR_TF_NOFACTOR}:
		if result_type == PRIMENET.AR_TF_FACTOR:
			factors = ar["factors"]
			buf += "M{0} has {1}factor{2}: {3} (TF:{4}-{5})".format(
				assignment.n,
				"a " if len(factors) == 1 else "",
				"s" if len(factors) != 1 else "",
				", ".join(factors),
				ar["bitlo"],
				ar["bithi"],
			)
		else:
			buf += "M{0} no factors from 2^{1} to 2^{2}{3}".format(
				assignment.n, ar["bitlo"], ar["bithi"], ", {0}".format(ar["security-code"]) if "security-code" in ar else ""
			)
	elif result_type in {PRIMENET.AR_P1_FACTOR, PRIMENET.AR_P1_NOFACTOR}:
		b1 = ar["B1"] if "B1" in ar else ar["b1"]
		b2 = None
		if "b2" in ar or "B2" in ar:
			b2 = ar["B2"] if "B2" in ar else ar["b2"]
		if result_type == PRIMENET.AR_P1_FACTOR:
			factors = ar["factors"]
			buf += "{0} has {1}factor{2}: {3} (P-1, B1={4}{5})".format(
				exponent_to_str(assignment),
				"a " if len(factors) == 1 else "",
				"s" if len(factors) != 1 else "",
				", ".join(factors),
				b1,
				", B2={0}{1}".format(b2, ", E={0}".format(ar["brent-suyama"]) if "brent-suyama" in ar else "")
				if b2 is not None
				else "",
			)
		else:
			buf += "{0} completed P-1, B1={1}{2}, {3}".format(
				exponent_to_str(assignment),
				b1,
				", B2={0}{1}".format(b2, ", E={0}".format(ar["brent-suyama"]) if "brent-suyama" in ar else "")
				if b2 is not None
				else "",
				ar.get("security-code", "-"),
			)
	elif result_type == PRIMENET.AR_CERT:
		buf += "{0} certification hash value {1}. {2},{3}{4}".format(
			exponent_to_str(assignment),
			ar["sha3-hash"],
			ar.get("security-code", "-"),
			"{0},".format(ar["shift-count"]) if "shift-count" in ar else "",
			ar.get("error-code", "0" * 8),
		)
	if assignment.uid:
		buf += ", AID: {0}".format(assignment.uid)

	no_report = False
	if result_type in {PRIMENET.AR_LL_PRIME, PRIMENET.AR_PRP_PRIME}:
		adigits = digits(assignment)
		no_report = options.no_report_100m and adigits >= 100000000
		if assignment.k == 1.0 and assignment.b == 2 and assignment.c == -1 and not is_known_mersenne_prime(assignment.n):
			string_rep = assignment_to_str(assignment)
			if not (config.has_option(SEC.PrimeNet, "SilentVictory") and config.getboolean(SEC.PrimeNet, "SilentVictory")):
				thread = threading.Thread(target=announce_prime_to_user, args=(string_rep, worktype))  # daemon=True
				thread.daemon = True
				thread.start()
			user_name = config.get(SEC.PrimeNet, "user_name")
			# Backup notification
			try:
				# "https://maker.ifttt.com/trigger/result_submitted/with/key/cIhVJKbcWgabfVaLuRjVsR"
				r = requests.post(
					"https://hook.us1.make.com/n16ouxwmfxts1o8154i9kfpqq1rwof53",
					json={"value1": user_name, "value2": buf, "value3": message},
					timeout=30,
				)
				text = r.text
			except RequestException as e:
				logging.exception("Backup notification failed: %s", e, exc_info=options.debug)
			else:
				adapter.debug("Backup notification: %s", text)
			if result_type == PRIMENET.AR_LL_PRIME:
				subject = "‼️ New Mersenne Prime!!! {0} is prime!".format(string_rep)
				temp = "Mersenne"
			else:
				subject = "❗ New Probable Prime!!! {0} is a probable prime!".format(string_rep)
				temp = "probable"
			if options.gpuowl:  # GpuOwl
				file = os.path.join(adir, "gpuowl.log")
				exp = os.path.join(adir, "*{0}".format(assignment.n))
				entries = glob.glob(
					os.path.join(exp, "{0}-[0-9]*.{1}".format(assignment.n, "ll" if result_type == PRIMENET.AR_LL_PRIME else "prp"))
				)
				if entries:
					savefile = next(
						entry
						for _, entry in sorted(
							(
								(tuple(map(int, os.path.splitext(os.path.basename(entry))[0].split("-"))), entry)
								for entry in entries
							),
							reverse=True,
						)
					)
				else:
					savefile = os.path.join(
						exp, "{0}.{1}owl".format(assignment.n, "ll." if result_type == PRIMENET.AR_LL_PRIME else "")
					)
			elif options.cudalucas:  # CUDALucas
				file = savefile = os.path.join(adir, "c{0}".format(assignment.n))
			else:  # Mlucas
				file = os.path.join(adir, "p{0}.stat".format(assignment.n))
				savefile = os.path.join(adir, "p{0}".format(assignment.n))
			send_msg(
				subject,
				"""This is an automated message sent by AutoPrimeNet.

User {0!r} (user ID: {1}) has allegedly found a new {2} prime on their {3!r} computer with the {4!r} GIMPS program!

Exponent: {5}, Decimal digits: {6:n}{7}

Result text format:

> {8}

Result JSON format:

> {9}

Below is the last up to 100 lines of the log file for {10} (the program may have moved on to a different exponent):

{11}

Attached is a zipped copy of the full log file and last savefile/checkpoint file for the user’s GIMPS program and the log file for AutoPrimeNet.

Exponent links: https://www.mersenne.org/M{12}, https://www.mersenne.ca/M{12}

AutoPrimeNet version: {13}
Requests/urllib3 library version: {14}/{15}
Python version: {16}
""".format(
					user_name,
					options.user_id,
					temp,
					computer,
					aprogram,
					string_rep,
					adigits,
					" ‼️" if adigits >= 100000000 else "",
					buf,
					message,
					PROGRAM["name"],
					"N/A" if options.cudalucas else tail(file),
					assignment.n,
					VERSION,
					requests.__version__,
					urllib3.__version__,
					platform.python_version(),
				),
				([] if options.cudalucas else [file]) + [savefile, logfile],
				cc=None if no_report or "known-factors" in ar else CCEMAILS,
				priority="1 (Highest)",
				azipfile="attachments.zip",
			)

	if not no_report:
		adapter.debug("Sending result: %r", sendline)
		adapter.info("Sending result to server: %s", buf)

	if result_type in {PRIMENET.AR_TF_FACTOR, PRIMENET.AR_P1_FACTOR}:
		for factor in map(int, ar["factors"]):
			adapter.info(
				"The %s factor %s has %s decimal digits and %g bits",
				"prime" if is_prime(factor) else "composite",
				factor,
				len(str(factor)),
				log2(factor),
			)
			if result_type == PRIMENET.AR_TF_FACTOR:
				if pow(2, assignment.n, factor) - 1:
					adapter.warning("Bad factor for M%s found: %s", assignment.n, factor)
			elif (int(assignment.k) * pow(assignment.b, assignment.n, factor) + assignment.c) % factor:
				adapter.warning("Bad factor for %s found: %s", exponent_to_str(assignment), factor)

	return ar, message, assignment, result_type, no_report


RESULTPATTERN = re.compile(r"Prime95|Program: E|Mlucas|CUDALucas v|CUDAPm1 v|gpuowl|prpll|mfakt[co]|cofact")


def submit_work(dirs, adapter, adir, cpu_num, tasks):
	"""Submits the results file to the PrimeNet server."""
	# A cumulative backup
	sentfile = os.path.join(adir, "results_sent.txt")
	results_sent = frozenset(readonly_list_file(sentfile))

	# Only submit completed work, i.e. the exponent must not exist in worktodo file any more
	# appended line by line, no lock needed
	resultsfile = os.path.join(adir, options.results_file)
	with LockFile(resultsfile):
		results = readonly_list_file(resultsfile)
		# EWM: Note that readonly_list_file does not need the file(s) to exist - nonexistent files simply yield 0-length rs-array entries.
		# remove nonsubmittable lines from list of possibles
		# if a line was previously submitted, discard
		results_send = [line for line in results if RESULTPATTERN.search(line) and line not in results_sent]

	if not results_send:
		adapter.debug("No new results in %r.", resultsfile)
		return
	length = len(results_send)
	adapter.debug("Found %s new result%s to report in %r", length, "s" if length != 1 else "", resultsfile)

	# Only for new results, to be appended to results_sent
	mersenne_ca_result_send = []
	accepted = 0
	rejected = {}
	failed = []
	ghzdays = []

	if config.has_option(SEC.Email, "PrimeNet_error_code_ignore"):
		error_code_ignore = config.get(SEC.Email, "PrimeNet_error_code_ignore")
		error_code_ignore = frozenset(map(int, error_code_ignore.split(",")) if error_code_ignore else ())
	else:
		error_code_ignore = frozenset((PRIMENET.ERROR_NO_ASSIGNMENT,))

	# EWM: Switch to one-result-line-at-a-time submission to support
	# error-message-on-submit handling:
	with io.open(sentfile, "a", encoding="utf-8") as file:
		for sendline in results_send:
			result = parse_result(adapter, adir, resultsfile, sendline)
			if result is not None:
				ar, message, assignment, result_type, no_report = result
				is_sent = False
				if no_report:
					file.write(sendline + "\n")
					is_sent = True
				elif assignment.k == 1.0 and assignment.b == 2 and assignment.n >= MAX_PRIMENET_EXP and assignment.c == -1:
					mersenne_ca_result_send.append((message, sendline))
					is_sent = True
				else:
					result = report_result(adapter, ar, message, assignment, result_type, tasks)
					if result is not None:
						file.write(sendline + "\n")
						ec, ghd = result
						if not ec:
							accepted += 1
						elif ec not in error_code_ignore:
							reason = "PrimeNet error {0} ({1})".format(ec, ERRORS.get(ec, "Unknown error code"))
							rejected.setdefault(reason, []).append(sendline)
						ghzdays.append(ghd)
						is_sent = True
					else:
						failed.append(sendline)

				if is_sent:
					if result_type in {PRIMENET.AR_TF_FACTOR, PRIMENET.AR_P1_FACTOR}:
						config.set(SEC.Internals, "RollingStartTime", str(0))
						adjust_rolling_average(dirs)
					else:
						rolling_average_work_unit_complete(adapter, adir, cpu_num, tasks, assignment)

		if accepted > 1:
			adapter.info("Submitted %s result%s to mersenne.org.", accepted, "s" if accepted != 1 else "")
			adapter.info("Total credit is %s GHz-days.", format(sum(ghzdays), "n"))

		# send all mersenne.ca results at once, to minimize server overhead
		if mersenne_ca_result_send:
			messages, sendlines = zip(*mersenne_ca_result_send)
			result = submit_mersenne_ca_results(adapter, messages)
			if result is not None:
				file.writelines(sendline + "\n" for sendline in sendlines)
				if result:
					rejected.update(result)
			else:
				failed.extend(sendlines)

	if rejected:
		length = sum(map(len, rejected.values()))
		send_msg(
			"🚫📤 Assignment result{0} rejected on {1}".format("s" if length != 1 else "", options.computer_id),
			"""{0:n} assignment result{1} rejected from the {2!r} file on your {3!r} computer (worker #{4}):

{5}

Below is the last up to 10 lines of the {6!r} log file for AutoPrimeNet:

{7}

If you believe this is a bug with AutoPrimeNet, please create an issue: https://github.com/tdulcet/AutoPrimeNet/issues
""".format(
				length,
				"s" if length != 1 else "",
				resultsfile,
				options.computer_id,
				cpu_num + 1,
				"\n".join(
					"> {0}:\n".format(reason) + "\n".join("> \t" + line for line in lines) for reason, lines in rejected.items()
				)
				if length <= 10
				else "\n".join(
					"> {0}: {1:n} result{2}".format(reason, len(lines), "s" if len(lines) != 1 else "")
					for reason, lines in rejected.items()
				)
				+ "\n> (See results attached)",
				logfile,
				tail(logfile, 10),
			),
			[(resultsfile, "\n".join(reason + ":\n" + "\n".join(lines) for reason, lines in rejected.items()).encode("utf-8"))]
			if length > 10
			else None,
			priority="2 (High)",
		)

	if failed:
		length = len(failed)
		send_msg(
			"❌📤 Failed to report assignment result{0} on {1}".format("s" if length != 1 else "", options.computer_id),
			"""Failed to report {0:n} assignment result{1} from the {2!r} file on your {3!r} computer (worker #{4}):

{5}

Below is the last up to 10 lines of the {6!r} log file for AutoPrimeNet:

{7}

If you believe this is a bug with AutoPrimeNet, please create an issue: https://github.com/tdulcet/AutoPrimeNet/issues
""".format(
				length,
				"s" if length != 1 else "",
				resultsfile,
				options.computer_id,
				cpu_num + 1,
				"\n".join("> " + line for line in failed) if length <= 10 else "> (See results attached)",
				logfile,
				tail(logfile, 10),
			),
			[(resultsfile, "\n".join(failed).encode("utf-8"))] if length > 10 else None,
			priority="2 (High)",
		)


def tf1g_unreserve_all(adapter, cpu_num, retry_count=0):
	"""Unreserve all TF1G assignments for a given worker."""
	guid = get_guid(config)
	if not options.user_id:
		adapter.error("Failed to unreserve TF1G exponents due to missing PrimeNet User ID.")
		return False
	adapter.info("Unreserving TF1G assignments.")
	retry = False
	try:
		r = session.post(
			mersenne_ca_baseurl + "tf1G.php",
			data={"gimps_login": options.user_id, "unreserve-all": options.user_id, "cpu": guid, "worker": cpu_num},
			timeout=180,
		)
		r.raise_for_status()
		result = r.json()
	except RequestException as e:
		logging.exception("%s", e, exc_info=options.debug)
		retry = True
	else:
		if "error" in result:
			adapter.error("Error during TF1G unreserve-all: %s", result["error"])
			return False
		if "unreserved_count" in result:
			adapter.info("Unreserved %s exponents from TF1G.", result["unreserved_count"])
			return True
		adapter.error("Error during TF1G unreserve-all, unexpected result:")
		adapter.error("%s", result)
	if retry:
		if retry_count >= 2:
			adapter.info("Retry count exceeded.")
			return False
		time.sleep(1 << retry_count)
		return tf1g_unreserve_all(adapter, cpu_num, retry_count + 1)
	return False


def unreserve_all(dirs):
	"""Unreserves all assignments in the given directories."""
	logging.info("Unreserving all assignments.")
	for i, adir in enumerate(dirs):
		adapter = logging.LoggerAdapter(logger, {"cpu_num": i} if options.dirs else None)
		cpu_num = i if options.dirs else options.cpu
		workfile = os.path.join(adir, options.worktodo_file)
		any_tf1g = tf1g_unreserved = False
		with LockFile(workfile):
			tasks = list(read_workfile(adapter, adir))
			submit_work(dirs, adapter, adir, cpu_num, tasks)
			assignments = OrderedDict(
				((assignment.uid, assignment.n), assignment) for assignment in tasks if isinstance(assignment, Assignment)
			).values()
			changed = False
			for assignment in assignments:
				tf1g = False
				if assignment.work_type == PRIMENET.WORK_TYPE_FACTOR and assignment.n >= MAX_PRIMENET_EXP:
					if not any_tf1g:
						tf1g_unreserved = tf1g_unreserve_all(adapter, cpu_num)
					any_tf1g = tf1g = True
				if tf1g_unreserved if tf1g else assignment_unreserve(adapter, assignment):
					tasks = [
						task
						for task in tasks
						if not isinstance(task, Assignment)
						or (task.uid != assignment.uid if assignment.uid else task.n != assignment.n)
					]
					changed = True
			if changed:
				write_workfile(adir, tasks)


def update_assignment(adapter, cpu_num, assignment, task):
	"""Update the assignment based on various conditions and options, potentially converting work types and adjusting bounds."""
	bounds = ("MIN", "MID", "MAX")
	changed = False
	if assignment.work_type == PRIMENET.WORK_TYPE_PRP and (
		options.convert_prp_to_ll or (not assignment.prp_dblchk and int(options.work_preference[cpu_num]) in CONVERT_DICT)
	):
		adapter.info("Converting from PRP to LL")
		assignment.work_type = PRIMENET.WORK_TYPE_DBLCHK if assignment.prp_dblchk else PRIMENET.WORK_TYPE_FIRST_LL
		assignment.pminus1ed = int(not assignment.tests_saved)
		changed = True
	if assignment.work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK} and options.convert_ll_to_prp:
		adapter.info("Converting from LL to PRP")
		assignment.tests_saved = float(not assignment.pminus1ed)
		assignment.prp_dblchk = assignment.work_type == PRIMENET.WORK_TYPE_DBLCHK
		assignment.work_type = PRIMENET.WORK_TYPE_PRP
		changed = True
	if options.tests_saved is not None and assignment.work_type in {
		PRIMENET.WORK_TYPE_FIRST_LL,
		PRIMENET.WORK_TYPE_DBLCHK,
		PRIMENET.WORK_TYPE_PRP,
		PRIMENET.WORK_TYPE_PFACTOR,
	}:
		redo = False
		if (
			options.tests_saved
			and options.pm1_multiplier is not None
			and (
				(assignment.work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK} and assignment.pminus1ed)
				or (assignment.work_type == PRIMENET.WORK_TYPE_PRP and not assignment.tests_saved)
			)
		):
			json = get_exponent(assignment.n)
			if json is not None and int(json["exponent"]) == assignment.n:
				actual = json["current"]["actual"]
				bound1 = actual["b1"] and int(actual["b1"])
				bound2 = actual["b2"] and int(actual["b2"])
				if bound1 and bound2:
					adapter.debug("Existing P-1 bounds are B1=%s, B2=%s", bound1, bound2)
					prob1, prob2 = pm1(assignment.n, assignment.sieve_depth, bound1, bound2)
					adapter.debug(
						"Chance of finding a factor was an estimated %s (%s + %s)",
						format(prob1 + prob2, "%"),
						format(prob1, ".3%"),
						format(prob2, ".3%"),
					)
					_, (midB1, midB2), _ = walk(assignment.n, assignment.sieve_depth)
					adapter.debug("Optimal P-1 bounds are B1=%s, B2=%s", midB1, midB2)
					p1, p2 = pm1(assignment.n, assignment.sieve_depth, midB1, midB2)
					adapter.debug(
						"Chance of finding a factor is an estimated %s (%s + %s) or a difference of %s (%s + %s)",
						format(p1 + p2, "%"),
						format(p1, ".3%"),
						format(p2, ".3%"),
						format(p1 + p2 - (prob1 + prob2), "%"),
						format(p1 - prob1, ".3%"),
						format(p2 - prob2, ".3%"),
					)
					if bound2 < midB2 * options.pm1_multiplier:
						adapter.info("Existing B2=%s < %s, redoing P-1", bound2, midB2 * options.pm1_multiplier)
						redo = True
		else:
			redo = True
		if redo:
			if assignment.work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK}:
				assignment.pminus1ed = int(not options.tests_saved)
			elif assignment.work_type in {PRIMENET.WORK_TYPE_PRP, PRIMENET.WORK_TYPE_PFACTOR}:
				assignment.tests_saved = options.tests_saved
			changed = True
	if options.pm1_bounds:
		add_bounds = False
		if not (options.cudalucas or options.gpuowl) and assignment.work_type == PRIMENET.WORK_TYPE_PFACTOR:
			adapter.info("Converting from Pfactor= to Pminus1=")
			assignment.work_type = PRIMENET.WORK_TYPE_PMINUS1
			add_bounds = True
		elif options.gpuowl and (
			(assignment.work_type == PRIMENET.WORK_TYPE_PRP and assignment.tests_saved)
			or assignment.work_type == PRIMENET.WORK_TYPE_PFACTOR
		):
			add_bounds = True
		if add_bounds:
			B1, B2 = walk(assignment.n, assignment.sieve_depth)[bounds.index(options.pm1_bounds)]
			adapter.info("Adding %s optimal P-1 bounds B1=%s, B2=%s to assignment", options.pm1_bounds, B1, B2)
			p1, p2 = pm1(assignment.n, assignment.sieve_depth, B1, B2)
			adapter.debug(
				"Chance of finding a factor is an estimated %s (%s + %s)",
				format(p1 + p2, "%"),
				format(p1, ".3%"),
				format(p2, ".3%"),
			)
			assignment.B1 = B1
			assignment.B2 = B2
			changed = True
	if changed:
		adapter.debug("Original assignment: %r", task)
		task = output_assignment(assignment)
	adapter.debug("New assignment: %r", task)
	return assignment, task


def register_assignment(adapter, cpu_num, assignment, retry_count=0):
	"""Register a new assignment with the PrimeNet server."""
	guid = get_guid(config)
	if guid is None:
		adapter.error("Cannot register assignment, the registration is not done")
		return None
	if assignment.k == 1.0 and assignment.b == 2 and assignment.n >= MAX_PRIMENET_EXP and assignment.c == -1:
		adapter.error("Cannot register assignment, exponent is larger than PrimeNet bounds")
		return None
	args = primenet_v5_bargs.copy()
	args["t"] = "ra"
	args["g"] = guid
	args["c"] = cpu_num
	args["w"] = assignment.work_type
	args["n"] = assignment.n
	if assignment.work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK}:
		args["sf"] = assignment.sieve_depth
		args["p1"] = assignment.pminus1ed
	elif assignment.work_type == PRIMENET.WORK_TYPE_PRP:
		args["A"] = "{0:.0f}".format(assignment.k)
		args["b"] = assignment.b
		args["C"] = assignment.c
		args["sf"] = assignment.sieve_depth
		args["saved"] = assignment.tests_saved
	elif assignment.work_type == PRIMENET.WORK_TYPE_FACTOR:
		args["sf"] = assignment.sieve_depth
		if assignment.factor_to:
			args["ef"] = assignment.factor_to
	elif assignment.work_type == PRIMENET.WORK_TYPE_PFACTOR:
		args["A"] = "{0:.0f}".format(assignment.k)
		args["b"] = assignment.b
		args["C"] = assignment.c
		args["sf"] = assignment.sieve_depth
		args["saved"] = assignment.tests_saved
	elif assignment.work_type == PRIMENET.WORK_TYPE_PMINUS1:
		args["A"] = "{0:.0f}".format(assignment.k)
		args["b"] = assignment.b
		args["C"] = assignment.c
		args["B1"] = assignment.B1
		if assignment.B2:
			args["B2"] = assignment.B2
	# elif assignment.work_type == PRIMENET.WORK_TYPE_CERT:
	retry = False
	adapter.info("Registering assignment: %s", exponent_to_text(assignment))
	result = send_request(guid, args)
	if result is None:
		retry = True
	else:
		rc = int(result["pnErrorResult"])
		if rc == PRIMENET.ERROR_OK:
			assignment.uid = result["k"]
			adapter.info("Assignment registered as: %s", assignment.uid)
			return assignment
		if rc == PRIMENET.ERROR_NO_ASSIGNMENT:
			pass
		elif rc == PRIMENET.ERROR_INVALID_ASSIGNMENT_TYPE:
			pass
		elif rc == PRIMENET.ERROR_INVALID_PARAMETER:
			pass
		elif rc == PRIMENET.ERROR_UNREGISTERED_CPU:
			register_instance()
			retry = True
		elif rc == PRIMENET.ERROR_STALE_CPU_INFO:
			register_instance(guid)
			retry = True
	if retry:
		if retry_count >= 2:
			adapter.info("Retry count exceeded.")
			return None
		time.sleep(1 << retry_count)
		return register_assignment(adapter, cpu_num, assignment, retry_count + 1)
	return None


def register_assignments(adapter, adir, cpu_num, tasks):
	"""Registers any assignments with the PrimeNet server."""
	registered_assignment = False
	changed = False
	for i, assignment in enumerate(tasks):
		if isinstance(assignment, Assignment) and not assignment.uid and not assignment.ra_failed:
			registered = register_assignment(adapter, cpu_num, assignment)
			if registered:
				assignment = registered
				registered_assignment = True
			else:
				assignment.ra_failed = True
			task = output_assignment(assignment)
			assignment, _ = update_assignment(adapter, cpu_num, assignment, task)
			tasks[i] = assignment
			changed = True
	if changed:
		write_workfile(adir, tasks)
	return registered_assignment


def register_exponents(dirs):
	"""Registers specific exponents by generating assignment lines and adding them to the work file."""
	wrapper = textwrap.TextWrapper(width=75)
	print(
		wrapper.fill(
			"This option is for advanced users who want to register specific exponents by helping to generate the assignment lines. Most users do not need to do this, as the program will automatically get assignments when needed."
		)
		+ "\n"
	)

	cpu_num = (
		(ask_int("Worker number", 1, 1, len(options.dirs)) - 1 if len(options.dirs) > 1 else 0) if options.dirs else options.cpu
	)
	adir = dirs[cpu_num if options.dirs else 0]
	workfile = os.path.join(adir, options.worktodo_file)
	adapter = logging.LoggerAdapter(logger, None)

	with LockFile(workfile), io.open(workfile, "a", encoding="utf-8") as file:
		while True:
			print("""\nUse the following values to select a worktype:
	2 - Trial factoring (mfaktc/mfakto only) (Factor=)
	3 - P-1 factoring with bounds (Mlucas only) (Pminus1=)
	4 - P-1 factoring (with bounds for GpuOwl only) (Pfactor=)
	100 - First time LL test (Test=)
	101 - Double-check LL test (DoubleCheck=)
	150 - First time PRP test (PRP=)
	151 - Double-check PRP test (PRPDC=)
""")

			while True:
				work_type = ask_int("Type of work", None, 0, 161)
				if work_type is not None and work_type in {
					PRIMENET.WORK_TYPE_FIRST_LL,
					PRIMENET.WORK_TYPE_DBLCHK,
					PRIMENET.WORK_TYPE_PRP,
					151,
					PRIMENET.WORK_TYPE_FACTOR,
					PRIMENET.WORK_TYPE_PFACTOR,
					PRIMENET.WORK_TYPE_PMINUS1,
				}:
					break

			while True:
				p = ask_int("Exponent to test", None, 2, 10000000000)
				if p is not None:
					if is_prime(p):
						break
					print("This number is not prime, there is no need to test it.")

			json = get_exponent(p)
			sieve_depth = factor_to = pminus1ed = tests_saved = None
			known_factors = []
			if json is not None and int(json["exponent"]) == p:
				actual = json["current"]["actual"]
				sieve_depth = int(actual["tf"])
				bound1 = actual["b1"] and int(actual["b1"])
				bound2 = actual["b2"] and int(actual["b2"])
				pminus1ed = bool(bound1) and bool(bound2)
				if "factors_prime" in json:
					known_factors = [int(factor["factor"]) for factor in json["factors_prime"]]
				print("\nThis exponent has been Trial Factored (TFed) to {0:n} bits".format(sieve_depth))
				if pminus1ed:
					print("Existing P-1 bounds are B1={0:n}, B2={1:n}\n".format(bound1, bound2))
				else:
					print("It has not yet been P-1 factored: B1={0}, B2={1}\n".format(bound1, bound2))
			if sieve_depth is None:
				print(
					"\n"
					+ wrapper.fill(
						"Unfortunately, the program was unable to automatically determine the TF bits and P-1 bounds for this exponent. It may be above the mersenne.ca exponent limit."
					)
				)
				print(
					"""Here are the links to find this information:
https://www.mersenne.org/M{0}
https://www.mersenne.ca/M{0}
""".format(p)
				)

			if work_type == PRIMENET.WORK_TYPE_FACTOR:
				sieve_depth = ask_int("Trial Factor (TF) starting bits", sieve_depth or 0, 0, 99)
				factor_to = ask_int("Trial Factor (TF) ending bits", max(factor_limit(p), sieve_depth + 1), sieve_depth, 99)
			else:
				sieve_depth = ask_float("Trial Factor (TF) bits", sieve_depth, 0, 99)

			if work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK}:
				pminus1ed = ask_yn("Has it been P-1 factored before?", pminus1ed)
			elif work_type not in {PRIMENET.WORK_TYPE_FACTOR, PRIMENET.WORK_TYPE_PMINUS1}:
				tests_saved = ask_float("Primality tests saved if factor is found", 0.0 if pminus1ed else 1.3, 0)

			if work_type == 151:
				prp_base = ask_int("PRP base", 3, 2)
				prp_residue_type = ask_int("PRP residue type", 1, 1, 5)

			B1 = B2 = 0
			if (
				options.gpuowl
				and ((work_type in {PRIMENET.WORK_TYPE_PRP, 151} and tests_saved) or work_type == PRIMENET.WORK_TYPE_PFACTOR)
			) or work_type == PRIMENET.WORK_TYPE_PMINUS1:
				print("\nOptimal P-1 bounds:")
				_, (midB1, midB2), _ = bounds = walk(p, sieve_depth)
				for (B1, B2), label in zip(bounds, ("MIN", "MID", "MAX")):
					p1, p2 = pm1(p, sieve_depth, B1, B2)
					print("\t{0}: B1={1:n}, B2={2:n}, Probability {3:%} ({4:.3%} + {5:.3%})".format(label, B1, B2, p1 + p2, p1, p2))
				print("For more information, see: {0}prob.php?exponent={1}\n".format(mersenne_ca_baseurl, p))

				B1 = ask_int("P-1 Bound #1", 0 if options.gpuowl else midB1, 100)
				B2 = ask_int("P-1 Bound #2", 0 if options.gpuowl else midB2, 0)

			factors = []
			if (work_type == 151 and prp_residue_type == 5) or work_type in {
				PRIMENET.WORK_TYPE_PRP,
				PRIMENET.WORK_TYPE_PFACTOR,
				PRIMENET.WORK_TYPE_PMINUS1,
			}:
				product = 1
				for i in count():
					while True:
						factor = (
							ask_int("Known factor #{0:n}".format(i + 1), known_factors[i], 2)
							if i < len(known_factors)
							else ask_int(
								"Known factor #{0:n} (leave blank {1})".format(i + 1, "if none" if not i else "to continue"),
								None,
								2,
							)
						)
						if factor is None:
							break
						if not is_prime(factor):
							print("Factor is not prime")
						elif pow(2, p, product * factor) - 1:
							print("Bad factor for M{0}".format(p))
						else:
							product *= factor
							break
					if factor is None:
						break
					factors.append(factor)

			if not ask_ok_cancel():
				break

			assignment = Assignment()
			assignment.k = 1.0
			assignment.b = 2
			assignment.n = p
			assignment.c = -1
			if work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK}:
				assignment.work_type = work_type
				assignment.sieve_depth = sieve_depth
				assignment.pminus1ed = int(pminus1ed)
			elif work_type in {PRIMENET.WORK_TYPE_PRP, 151}:
				assignment.prp_dblchk = work_type == 151
				assignment.work_type = PRIMENET.WORK_TYPE_PRP
				assignment.B1 = B1
				assignment.B2 = B2
				assignment.sieve_depth = sieve_depth
				assignment.tests_saved = tests_saved
				if work_type == 151:
					assignment.prp_base = prp_base
					assignment.prp_residue_type = prp_residue_type
				assignment.known_factors = factors
			elif work_type == PRIMENET.WORK_TYPE_FACTOR:
				assignment.work_type = work_type
				assignment.sieve_depth = sieve_depth
				assignment.factor_to = factor_to
			elif work_type == PRIMENET.WORK_TYPE_PFACTOR:
				assignment.work_type = work_type
				assignment.B1 = B1
				assignment.B2 = B2
				assignment.sieve_depth = sieve_depth
				assignment.tests_saved = tests_saved
				assignment.known_factors = factors
			elif work_type == PRIMENET.WORK_TYPE_PMINUS1:
				assignment.work_type = work_type
				assignment.B1 = B1
				assignment.B2 = B2
				assignment.sieve_depth = sieve_depth
				assignment.known_factors = factors

			task = output_assignment(assignment)
			print("\nAdding assignment {0!r} to the {1!r} file\n".format(task, workfile))
			file.write(task + "\n")

			if not ask_yn("Do you want to register another exponent?", False):
				break

		tasks = list(read_workfile(adapter, adir))
		register_assignments(adapter, adir, cpu_num, tasks)


def tf1g_fetch(adapter, adir, cpu_num, max_assignments=None, max_ghd=None, recover=False, recover_all=False, retry_count=0):
	"""Fetches TF1G assignments from mersenne.ca with optional recovery."""
	guid = get_guid(config)
	data = {"gimps_login": options.user_id}
	if not recover_all:
		data.update({"cpu": guid, "worker": cpu_num})
	if recover or recover_all:
		adapter.info("Recovering TF1G assignments")
		data["myassignments"] = 1
	else:
		stages = get_stages_mfaktx_ini(adapter, adir)
		adapter.info(
			"Getting %s%s TF1G assignments from mersenne.ca, stages = %s",
			format(max_ghd or max_assignments, "n"),
			" GHz-days of" if max_ghd else "",
			stages,
		)
		data.update({
			"min_exponent": options.min_exp,
			"max_exponent": options.max_exp,
			"tf_min": options.min_bit,
			"tf_limit": options.max_bit,
			"max_ghd": max_ghd,
			"max_assignments": None if max_ghd else max_assignments,
			"download_worktodo": 1,
			"stages": stages,
		})
		if config.has_option(SEC.PrimeNet, "tf1g_biggest"):
			data["biggest"] = int(config.getboolean(SEC.PrimeNet, "tf1g_biggest"))
	retry = False
	try:
		r = session.post(mersenne_ca_baseurl + "tf1G.php", data=data, timeout=180, stream=True)
		r.raise_for_status()
		tests = []
		for task in r.iter_lines(decode_unicode=True):
			if task:
				test = parse_assignment(task)
				if test is None:
					adapter.error("Invalid assignment %r", task)
					tests.append(task)
				else:
					adapter.info("Got assignment: %r", exponent_to_text(test))
					tests.append(test)
	except RequestException as e:
		logging.exception("%s", e, exc_info=options.debug)
		retry = True
	else:
		if recover or recover_all:
			adapter.debug("Recovered %s TF1G assignment%s from mersenne.ca", len(tests), "s" if len(tests) != 1 else "")
		else:
			adapter.debug("Fetched %s TF1G assignment%s from mersenne.ca", len(tests), "s" if len(tests) != 1 else "")
		return tests
	if retry:
		if retry_count >= 2:
			adapter.info("Retry count exceeded.")
			return []
		time.sleep(1 << retry_count)
		return tf1g_fetch(adapter, adir, cpu_num, max_assignments, max_ghd, recover, recover_all, retry_count + 1)
	return []


def recover_assignments(dirs, recover_all=False):
	"""Recovers assignments from the PrimeNet server."""
	guid = get_guid(config)
	if guid is None:
		logging.error("Cannot recover assignments, the registration is not done")
		return
	for i, adir in enumerate(dirs):
		adapter = logging.LoggerAdapter(logger, {"cpu_num": i} if options.dirs else None)
		cpu_num = i if options.dirs else options.cpu
		workfile = os.path.join(adir, options.worktodo_file)
		with LockFile(workfile):
			tasks = list(read_workfile(adapter, adir))
			submit_work(dirs, adapter, adir, cpu_num, tasks)
			num_to_get = get_assignment(adapter, cpu_num, 0, recover_all=recover_all)
			if num_to_get is None:
				adapter.error("Unable to determine the number of assignments to recover")
				return
			adapter.info("Recovering %s PrimeNet assignment%s", num_to_get, "s" if num_to_get != 1 else "")
			tests = []
			for j in range(1, num_to_get + 1):
				test = get_assignment(adapter, cpu_num, j, recover_all=recover_all)
				if test is None:
					break
				task = output_assignment(test)
				test, _ = update_assignment(adapter, cpu_num, test, task)
				tests.append(test)

			if options.min_exp and options.min_exp >= MAX_PRIMENET_EXP and (not recover_all or not cpu_num):
				for test in tf1g_fetch(adapter, adir, cpu_num, recover=True, recover_all=recover_all):
					if isinstance(test, Assignment):
						task = output_assignment(test)
						test, _ = update_assignment(adapter, cpu_num, test, task)
					tests.append(test)

			if len(tests) > 1:
				adapter.info("Recovered %s assignment%s", len(tests), "s" if len(tests) != 1 else "")
			write_workfile(adir, tests)

	# As of early 2018, here is the full list of assignment-type codes supported by the Primenet server; Mlucas
	# v20 (and thus this script) supports only the subset of these indicated by an asterisk in the left column.
	# Supported assignment types may be specified via either their PrimeNet number code or the listed Mnemonic:
	# 			Worktype:
	# Code		Mnemonic			Description
	# ----	-----------------	-----------------------
	#    0						Whatever makes the most sense
	#    1						Trial factoring to low limits
	# *  2						Trial factoring
	# *  4	Pfactor				P-1 factoring
	#    5						ECM for first factor on Mersenne numbers
	#    6						ECM on Fermat numbers
	#    8						ECM on mersenne cofactors
	#   12                      Trial factoring GPU
	# *100	SmallestAvail		Smallest available first-time tests
	# *101	DoubleCheck			Double-checking
	# *102	WorldRecord			World record primality tests
	# *104	100Mdigit			100M digit number to LL test (not recommended)
	# *150	SmallestAvailPRP	First time PRP tests (Gerbicz)
	# *151	DoubleCheckPRP		Doublecheck PRP tests (Gerbicz)
	# *152	WorldRecordPRP		World record sized numbers to PRP test (Gerbicz)
	# *153	100MdigitPRP		100M digit number to PRP test (Gerbicz)
	#  160						PRP on Mersenne cofactors
	#  161						PRP double-checks on Mersenne cofactors


def send_progress(adapter, cpu_num, assignment, percent, stage, time_left, now, fftlen, retry_count=0):
	"""Sends the expected completion date for a given assignment to the PrimeNet server."""
	guid = get_guid(config)
	if guid is None:
		adapter.error("Cannot send progress, the registration is not done")
		return None
	if assignment.n >= MAX_PRIMENET_EXP:
		# adapter.debug("Cannot send progress, exponent larger than PrimeNet bounds")
		return None
	if not assignment.uid:
		return None
	# Assignment Progress fields:
	# g= the machine's GUID (32 chars, assigned by Primenet on 1st-contact from a given machine, stored in 'guid=' entry of prime.ini file of rundir)
	args = primenet_v5_bargs.copy()
	args["t"] = "ap"  # update compute command
	args["g"] = guid
	# k= the assignment ID (32 chars, follows '=' in Primenet-generated workfile entries)
	args["k"] = assignment.uid
	# p= progress in %-done, 4-char format = xy.z
	args["p"] = "{0:.4f}".format(percent * 100)
	# d= when the client is expected to check in again (in seconds ... )
	args["d"] = options.hours_between_checkins * 60 * 60
	# e= the ETA of completion in seconds, if unknown, just put 1 week
	args["e"] = int(time_left) if time_left is not None else 7 * 24 * 60 * 60
	# c= the worker thread of the machine
	args["c"] = cpu_num
	# stage= LL in this case, although an LL test may be doing TF or P-1 work
	# first so it's possible to be something besides LL
	if stage:
		args["stage"] = stage
	if fftlen:
		args["fftlen"] = fftlen
	retry = False
	delta = timedelta(seconds=time_left)
	adapter.info(
		"Sending expected completion date for %s: %12s (%s)", exponent_to_str(assignment), delta, (now + delta).strftime("%c")
	)
	result = send_request(guid, args)
	if result is None:
		# Try again
		retry = True
	else:
		rc = int(result["pnErrorResult"])
		if rc == PRIMENET.ERROR_OK:
			# adapter.debug("Update correctly sent to server")
			pass
		elif rc == PRIMENET.ERROR_INVALID_ASSIGNMENT_KEY:
			# TODO: Delete assignment from workfile
			pass
		elif rc == PRIMENET.ERROR_WORK_NO_LONGER_NEEDED:
			# TODO: Delete assignment from workfile
			pass
		elif rc == PRIMENET.ERROR_UNREGISTERED_CPU:
			register_instance()
			retry = True
		elif rc == PRIMENET.ERROR_STALE_CPU_INFO:
			register_instance(guid)
			retry = True
		elif rc == PRIMENET.ERROR_SERVER_BUSY:
			retry = True
	if retry:
		if retry_count >= 2:
			adapter.info("Retry count exceeded.")
			return None
		time.sleep(1 << retry_count)
		return send_progress(adapter, cpu_num, assignment, percent, stage, time_left, now, fftlen, retry_count + 1)
	return None


def update_progress(adapter, cpu_num, assignment, progress, msec_per_iter, p, now, cur_time_left, checkin=True):
	"""Update the progress of a given assignment."""
	if not assignment:
		return None
	iteration, _, stage, _pct_complete, fftlen, bits, s2 = progress
	percent, time_left, msec_per_iter = compute_progress(assignment, iteration, msec_per_iter, p, bits, s2)
	adapter.debug(
		"%s is %s done (%s / %s)",
		assignment.n,
		format(percent, ".4%"),
		format(iteration, "n"),
		format(
			s2
			or bits
			or (
				assignment.n
				if assignment.work_type == PRIMENET.WORK_TYPE_PRP
				else assignment.cert_squarings
				if assignment.work_type == PRIMENET.WORK_TYPE_CERT
				else tf_ghd_credit(assignment.n, int(assignment.sieve_depth), int(assignment.factor_to))
				if assignment.work_type == PRIMENET.WORK_TYPE_FACTOR
				else assignment.n - 2
			),
			"n",
		),
	)
	if stage is None and percent > 0:
		if assignment.work_type in {PRIMENET.WORK_TYPE_FIRST_LL, PRIMENET.WORK_TYPE_DBLCHK}:
			stage = "LL"
		elif assignment.work_type == PRIMENET.WORK_TYPE_PRP:
			stage = "PRP"
		elif assignment.work_type == PRIMENET.WORK_TYPE_FACTOR:
			if int(assignment.factor_to) == int(assignment.sieve_depth) + 1:
				stage = "TF{0:.0f}".format(assignment.sieve_depth)
			else:
				stage = "TF{0:.0f}-{1:.0f}".format(assignment.sieve_depth, assignment.factor_to)
		elif assignment.work_type == PRIMENET.WORK_TYPE_CERT:
			stage = "CERT"
	if time_left is None:
		cur_time_left += 7 * 24 * 60 * 60
		adapter.debug("Finish cannot be estimated, using 7 days")
	else:
		cur_time_left += time_left
		delta = timedelta(seconds=time_left)
		adapter.debug("Finish estimated in %s (using %g ms/iter estimation)", delta, msec_per_iter)
	if checkin:
		send_progress(adapter, cpu_num, assignment, percent, stage, cur_time_left, now, fftlen)
	return percent, cur_time_left


def get_assignments(adapter, adir, cpu_num, progress, tasks):
	"""Get new assignments from the PrimeNet server."""
	if config.has_option(SEC.PrimeNet, "QuitGIMPS") and config.getboolean(SEC.PrimeNet, "QuitGIMPS"):
		return
	now = datetime.now()
	workfile = os.path.join(adir, options.worktodo_file)
	assignments = OrderedDict(
		((assignment.uid, assignment.n), assignment) for assignment in tasks if isinstance(assignment, Assignment)
	).values()
	_percent = cur_time_left = msec_per_iter = p = None
	if progress is not None:
		_percent, cur_time_left, msec_per_iter, p = progress  # unpack update_progress output
	section = "Worker #{0}".format(cpu_num + 1) if options.num_workers > 1 else SEC.Internals
	if (
		msec_per_iter is None
		and p is None
		and config.has_option(section, "msec_per_iter")
		and config.has_option(section, "exponent")
	):
		# get speed from .ini in case of no assignments
		msec_per_iter = config.getfloat(section, "msec_per_iter")
		p = config.getint(section, "exponent")
	num_cache = options.num_cache
	if not num_cache:
		if cur_time_left is None:
			num_cache = (
				(250 if options.min_exp and options.min_exp >= MAX_PRIMENET_EXP else 20) if options.mfaktc or options.mfakto else 1
			)
		else:
			num_cache = 1
		adapter.debug("The num_cache option was not set, defaulting to %s assignment%s", num_cache, "s" if num_cache != 1 else "")
	else:
		num_cache += 1
	num_existing = len(assignments)
	if num_existing > num_cache:
		adapter.debug(
			"Number of existing assignments (%s) in %r is greater than num_cache (%s), so num_cache increased to %s",
			num_existing,
			workfile,
			num_cache,
			num_existing,
		)
		num_cache = num_existing
	if cur_time_left is None:
		cur_time_left = 0
		adapter.debug(
			"Unable to estimate time left for current assignment%s, so only getting %s for now, instead of %s day%s of work",
			"s" if num_existing != 1 else "",
			num_cache,
			options.days_of_work,
			"s" if options.days_of_work != 1 else "",
		)

	if config.has_option(SEC.PrimeNet, "MaxExponents"):
		amax = config.getint(SEC.PrimeNet, "MaxExponents")
	else:
		amax = (
			(10000 if options.min_exp and options.min_exp >= MAX_PRIMENET_EXP else 1000) if options.mfaktc or options.mfakto else 15
		)
	days_work = timedelta(days=options.days_of_work)
	new_tasks = []
	while True:
		if num_cache <= num_existing and cur_time_left:
			time_left = timedelta(seconds=cur_time_left)
			if time_left <= days_work:
				num_cache += 1
				adapter.debug(
					"Time left (%s) is less than the days of work option (%s), so num_cache increased by one to %s",
					time_left,
					days_work,
					num_cache,
				)
			else:
				adapter.debug(
					"Time left (%s) is greater than the days of work option (%s), so num_cache has not been changed",
					time_left,
					days_work,
				)

		if amax < num_cache:
			adapter.info(
				"num_cache (%s) is greater than config option MaxExponents (%s), so num_cache decreased to %s",
				num_cache,
				amax,
				amax,
			)
			num_cache = amax

		if num_cache <= num_existing:
			adapter.debug("%s ≥ %s assignments already in %r, not getting new work", num_existing, num_cache, workfile)
			if cur_time_left and options.min_exp and options.min_exp >= MAX_PRIMENET_EXP:
				adapter.info("Estimated time to complete queued work is %s, days of work requested is %s", time_left, days_work)
			if not new_tasks:
				return
			break
		num_to_get = num_cache - num_existing
		adapter.debug(
			"Found %s < %s assignments in %r, getting %s new assignment%s",
			num_existing,
			num_cache,
			workfile,
			num_to_get,
			"s" if num_to_get != 1 else "",
		)

		if (
			options.min_exp
			and options.min_exp >= MAX_PRIMENET_EXP
			and work_preference[cpu_num] in {PRIMENET.WP_FACTOR, PRIMENET.WP_GPU_FACTOR}
		):
			ghd_to_request = None
			if msec_per_iter is not None:
				ghd_to_request = max(10, ((options.days_of_work * 24 * 60 * 60) - cur_time_left) * 1000 / msec_per_iter)
			assignments = tf1g_fetch(adapter, adir, cpu_num, num_to_get, ghd_to_request)
		else:
			assignments = []
			assignment = get_assignment(adapter, cpu_num, min_exp=options.min_exp, max_exp=options.max_exp)
			if assignment is not None:
				assignments.append(assignment)

		num_fetched = len(assignments)
		if not assignments:
			break
		with io.open(workfile, "a", encoding="utf-8") as file:
			for i, assignment in enumerate(assignments):
				if isinstance(assignment, Assignment):
					new_task = output_assignment(assignment)
					assignment, new_task = update_assignment(adapter, cpu_num, assignment, new_task)
					assignments[i] = assignment
					result = get_progress_assignment(adapter, adir, assignment)
					_percent, cur_time_left = update_progress(
						adapter, cpu_num, assignment, result, msec_per_iter, p, now, cur_time_left
					)
				else:
					new_task = assignment
				new_tasks.append(new_task)
				file.write(new_task + "\n")
		tasks.extend(assignments)
		num_existing += num_fetched

	if len(new_tasks) > 1:
		adapter.info("Fetched %s assignment%s", len(new_tasks), "s" if len(new_tasks) != 1 else "")
	if len(tasks) <= 5:
		output_status((adir,), cpu_num)
	if num_fetched < num_to_get:
		adapter.error(
			"Failed to get requested number of new assignments, %s requested, %s successfully retrieved", num_to_get, num_fetched
		)
		send_msg(
			"❌📥 Failed to get new assignments on {0}".format(options.computer_id),
			"""Failed to get new assignments for the {0!r} file on your {1!r} computer (worker #{2}).

Below is the last up to 10 lines of the {3!r} log file for AutoPrimeNet:

{4}

If you believe this is a bug with AutoPrimeNet, please create an issue: https://github.com/tdulcet/AutoPrimeNet/issues
""".format(workfile, options.computer_id, cpu_num + 1, logfile, tail(logfile, 10)),
			priority="2 (High)",
		)


def update_progress_all(adapter, adir, cpu_num, last_time, tasks, checkin=True):
	"""Update the progress of all the assignments in the workfile."""
	if not tasks:
		return None  # don't update if no worktodo
	# Treat the first assignment. Only this one is used to save the msec_per_iter
	# The idea is that the first assignment is having a .stat file with correct values
	# Most of the time, a later assignment would not have a .stat file to obtain information,
	# but if it has, it may come from an other computer if the user moved the files, and so
	# it doesn't have relevant values for speed estimation.
	# Using msec_per_iter from one p to another is a good estimation if both p are close enough
	# if there is big gap, it will be other or under estimated.
	# Any idea for a better estimation of assignment duration when only p and
	# type (LL or PRP) is known ?
	now = datetime.now()
	assignments = iter(
		OrderedDict(
			((assignment.uid, assignment.n), assignment) for assignment in tasks if isinstance(assignment, Assignment)
		).values()
	)
	assignment = next(assignments, None)
	if assignment is None:
		return None
	result = get_progress_assignment(adapter, adir, assignment)
	msec_per_iter = result[1]
	p = assignment.n
	section = "Worker #{0}".format(cpu_num + 1) if options.num_workers > 1 else SEC.Internals
	modified = True
	file = os.path.join(
		adir,
		"M{0}.ckp".format(p)
		if options.mfaktc or options.mfakto
		else "c{0}".format(p)
		if options.cudalucas
		else "gpuowl.log"
		if options.gpuowl
		else "p{0}.stat".format(p),
	)
	if os.path.exists(file) and last_time is not None:
		mtime = os.path.getmtime(file)
		date = datetime.fromtimestamp(mtime)
		if last_time >= mtime:
			adapter.warning(
				"STALL DETECTED: Log/Save file %r has not been modified since the last progress update (%s)",
				file,
				date.strftime("%c"),
			)
			if not config.has_option(section, "stalled"):
				logfile = (
					os.path.join(adir, "mfaktc.log" if options.mfaktc else "mfakto.log")
					if options.mfaktc or options.mfakto
					else file
				)
				send_msg(
					"⚠️ {0} on {1} has stalled".format(PROGRAM["name"], options.computer_id),
					"""The {0} program on your {1!r} computer (worker #{2}) has not made any progress for {3} ({4:%c}).

Below is the last up to 100 lines of the {5!r} log file:

{6}

This program will alert you when it has resumed.
""".format(
						PROGRAM["name"],
						options.computer_id,
						cpu_num + 1,
						now - date,
						date,
						logfile,
						"N/A" if options.cudalucas else tail(logfile),
					),
					priority="1 (Highest)",
				)
				config.set(section, "stalled", str(mtime))
			modified = False
		elif config.has_option(section, "stalled"):
			stalled = datetime.fromtimestamp(config.getfloat(section, "stalled"))
			send_msg(
				"✔️ {0} on {1} has resumed".format(PROGRAM["name"], options.computer_id),
				"""The {0} program on your {1!r} computer (worker #{2}) has resumed making progress.

It was stalled for {3}.
""".format(PROGRAM["name"], options.computer_id, cpu_num + 1, date - stalled),
			)
			config.remove_option(section, "stalled")
	checkin = checkin and modified
	if msec_per_iter is not None:
		config.set(section, "msec_per_iter", "{0:f}".format(msec_per_iter))
		config.set(section, "exponent", str(p))
	elif config.has_option(section, "msec_per_iter") and config.has_option(section, "exponent"):
		# If not speed available, get it from the prime.ini file
		msec_per_iter = config.getfloat(section, "msec_per_iter")
		p = config.getint(section, "exponent")
	# Do the other assignment accumulating the time_lefts
	cur_time_left = 0
	percent, cur_time_left = update_progress(adapter, cpu_num, assignment, result, msec_per_iter, p, now, cur_time_left, checkin)

	for assignment in assignments:
		result = get_progress_assignment(adapter, adir, assignment)
		percent, cur_time_left = update_progress(
			adapter, cpu_num, assignment, result, msec_per_iter, p, now, cur_time_left, checkin
		)
	return percent, cur_time_left, msec_per_iter, p


def ping_server(ping_type=1):
	"""Sends a ping to the PrimeNet server to check connectivity."""
	guid = get_guid(config)
	args = primenet_v5_bargs.copy()
	args["t"] = "ps"
	args["q"] = ping_type
	logging.info("Contacting PrimeNet Server.")
	result = send_request(guid, args)
	if result is None:
		pass
	else:
		rc = int(result["pnErrorResult"])
		if rc == PRIMENET.ERROR_OK:
			return result["r"]
	return None


def is_pyinstaller():
	"""Check if the script is running as a PyInstaller bundle."""
	# Adapted from: https://pyinstaller.org/en/stable/runtime-information.html
	return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


#######################################################################################################
#
# Start main program here
#
#######################################################################################################


parser = optparse.OptionParser(
	usage="%prog [options]\nUse -h/--help to see all options\nUse --setup to configure this instance of the program",
	version="%prog " + VERSION,
	description="This program will automatically get and register assignments, report assignment progress and results, upload proof files to and download certification starting values from PrimeNet for the Mlucas, GpuOwl/PRPLL, CUDALucas, mfaktc and mfakto GIMPS programs. It can get assignments and report results to mersenne.ca for exponents above the PrimeNet limit of 1G. It also saves its configuration to a 'prime.ini' file by default, so it is only necessary to give most of the arguments once. The first time it is run, it will register the current Mlucas/GpuOwl/PRPLL/CUDALucas/mfaktc/mfakto instance with PrimeNet (see the Registering Options below). Then, it will report assignment results, get assignments and upload any proof files to PrimeNet on the --timeout interval, or only once if --timeout is 0. It will additionally report the progress on the --checkin interval.",
)

# options not saved to prime.ini
parser.add_option(
	"-d",
	"--debug",
	action="count",
	dest="debug",
	default=0,
	help="Output detailed information. Provide multiple times for even more verbose output.",
)
parser.add_option(
	"-w",
	"--workdir",
	dest="workdir",
	default=os.curdir,
	help="Working directory with the configuration file from this program, Default: %default (current directory)",
)
parser.add_option(
	"-D",
	"--dir",
	action="append",
	dest="dirs",
	help="Directories relative to --workdir with the work and results files from the GIMPS program. Provide once for each worker. It automatically sets the --cpu-num option for each directory.",
)

# all other options are saved to prime.ini
parser.add_option("-i", "--work-file", dest="worktodo_file", default="worktodo.txt", help="Work file filename, Default: '%default'")
parser.add_option(
	"-r",
	"--results-file",
	dest="results_file",
	help="Results file filename, Default: 'results.json.txt' for mfaktc/mfakto or 'results.txt' otherwise",
)
parser.add_option("-L", "--logfile", dest="logfile", default="autoprimenet.log", help="Log file filename, Default: '%default'")
parser.add_option(
	"-l", "--config-file", dest="localfile", default="prime.ini", help="Local configuration file filename, Default: '%default'"
)
parser.add_option(
	"--archive-proofs", dest="archive_dir", help="Directory to archive PRP proof files after upload, Default: %default"
)
parser.add_option(
	"-u",
	"--username",
	dest="user_id",
	help="GIMPS/PrimeNet User ID. Create a GIMPS/PrimeNet account: https://www.mersenne.org/update/. If you do not want a PrimeNet account, you can use ANONYMOUS.",
)
parser.add_option("-p", "--password", help=optparse.SUPPRESS_HELP)

# -t is reserved for timeout, instead use -T for assignment-type preference:
parser.add_option(
	"-T",
	"--workpref",
	action="append",
	dest="work_preference",
	default=[],
	help="""Work preference, Default: {0}. Supported work preferences:
2 (Trial factoring),
4 (P-1 factoring),
12 (Trial factoring GPU),
100 (First time LL tests),
101 (Double-check LL tests),
102 (World record LL tests),
104 (100M digit LL tests),
106 (Double-check LL tests with zero shift count),
150 (First time PRP tests),
151 (Double-check PRP tests),
152 (World record PRP tests),
153 (100M digit PRP tests),
154 (Smallest available first time PRP that needs P-1 factoring),
155 (Double-check using PRP with proof),
156 (Double-check using PRP with proof and nonzero shift count),
160 (First time PRP on Mersenne cofactors),
161 (Double-check PRP on Mersenne cofactors).
Provide once to use the same work preference for all workers or once for each worker to use different work preferences. Not all worktypes are supported by all the GIMPS programs.""".format(
		PRIMENET.WP_PRP_FIRST
	),
)
parser.add_option(
	"--cert-work",
	action="store_true",
	dest="cert_work",
	help="Get PRP proof certification work, Default: %default. Currently only supported by PRPLL.",
)
parser.add_option(
	"--cert-work-limit",
	dest="cert_cpu_limit",
	type="int",
	default=10,
	help="PRP proof certification work limit in percentage of CPU or GPU time, Default: %default%. Requires the --cert-work option.",
)
parser.add_option(
	"--min-exp",
	dest="min_exp",
	type="int",
	help="Minimum exponent to get from PrimeNet or TF1G (2 - 9,999,999,999). TF1G assignments are supported by setting this flag to 1,000,000,000 or above.",
)
parser.add_option("--max-exp", dest="max_exp", type="int", help="Maximum exponent to get from PrimeNet or TF1G (2 - 9,999,999,999)")

parser.add_option("--min-bit", dest="min_bit", type="int", help="Minimum bit level of TF assignments to get from PrimeNet or TF1G")
parser.add_option("--max-bit", dest="max_bit", type="int", help="Maximum bit level of TF assignments to get from PrimeNet or TF1G")

parser.add_option("-m", "--mlucas", action="store_true", help="Get assignments for Mlucas.")
parser.add_option("-g", "--gpuowl", action="store_true", help="Get assignments for GpuOwl.")
parser.add_option(
	"--prpll",
	action="store_true",
	help="Get assignments for PRPLL. PRPLL is not PrimeNet server compatible and is thus not yet fully supported.",
)
parser.add_option("--cudalucas", action="store_true", help="Get assignments for CUDALucas.")
parser.add_option("--mfaktc", action="store_true", help="Get assignments for mfaktc.")
parser.add_option("--mfakto", action="store_true", help="Get assignments for mfakto.")
parser.add_option("--prime95", action="store_true", help=optparse.SUPPRESS_HELP)
parser.add_option(
	"--num-workers", dest="num_workers", type="int", default=1, help="Number of workers (CPU Cores/GPUs), Default: %default"
)
parser.add_option(
	"-c",
	"--cpu-num",
	dest="cpu",
	type="int",
	default=0,
	help="CPU core or GPU number to get assignments for, Default: %default. Deprecated in favor of the --dir option.",
)
parser.add_option(
	"-n",
	"--num-cache",
	dest="num_cache",
	type="int",
	default=0,
	help="Number of assignments to cache, Default: %default. Deprecated in favor of the --days-work option.",
)
parser.add_option(
	"-W",
	"--days-work",
	dest="days_of_work",
	type="float",
	help="Days of work to queue ((0-180] days), Default: 1 day for mfaktc/mfakto or 3 days otherwise. Increases num_cache when the time left for all assignments is less than this number of days.",
)
parser.add_option(
	"--force-pminus1",
	dest="tests_saved",
	type="float",
	help="Force P-1 factoring before LL/PRP tests and/or change the default PrimeNet PRP and P-1 tests_saved value.",
)
parser.add_option(
	"--pminus1-threshold",
	dest="pm1_multiplier",
	type="float",
	help="Retry the P-1 factoring before LL/PRP tests only if the existing P-1 bounds are less than the target bounds (as listed on mersenne.ca) times this threshold/multiplier. Requires the --force-pminus1 option.",
)
parser.add_option(
	"--force-pminus1-bounds",
	dest="pm1_bounds",
	choices=("MIN", "MID", "MAX"),
	help="Force using the 'MIN', 'MID' or 'MAX' optimal P-1 bounds (as listed on mersenne.ca) for P-1 tests. For Mlucas, this will rewrite Pfactor= assignments to Pminus1=. For GpuOwl, this will use a nonstandard Pfactor= format to add the bounds. Can be used in combination with the --force-pminus1 option.",
)
parser.add_option(
	"--convert-ll-to-prp",
	action="store_true",
	dest="convert_ll_to_prp",
	help="Convert all LL assignments to PRP. This is for use when registering assignments.",
)
parser.add_option(
	"--convert-prp-to-ll",
	action="store_true",
	dest="convert_prp_to_ll",
	help="Convert all PRP assignments to LL. This is automatically enabled for first time PRP assignments when the --workpref option is for a first time LL worktype.",
)
parser.add_option(
	"--no-report-100m",
	action="store_true",
	dest="no_report_100m",
	help="Do not report any prime results for exponents greater than or equal to 100 million digits. You must setup another method to notify yourself, such as setting the notification options below.",
)

parser.add_option(
	"--checkin",
	dest="hours_between_checkins",
	type="int",
	default=1,
	help="Hours to wait between sending assignment progress and expected completion dates (1-168 hours), Default: %default hours.",
)
parser.add_option(
	"-t",
	"--timeout",
	dest="timeout",
	type="int",
	default=60 * 60,
	help="Seconds to wait between updates, Default: %default seconds (1 hour). Users with slower internet may want to set a larger value to give time for any PRP proof files to upload. Use 0 to update once and exit.",
)
parser.add_option(
	"-s",
	"--status",
	action="store_true",
	dest="status",
	help="Output a status report and any expected completion dates for all assignments and exit.",
)
parser.add_option(
	"--upload-proofs",
	action="store_true",
	dest="proofs",
	help="Report assignment results, upload all PRP proofs and exit. Requires PrimeNet User ID.",
)
parser.add_option(
	"--recover",
	action="store_true",
	dest="recover",
	help="Report assignment results, recover all assignments and exit. This will overwrite any existing work files.",
)
parser.add_option(
	"--recover-all",
	action="store_true",
	dest="recover_all",
	help="The same as --recover, except for PrimeNet it will additionally recover expired assignments and for mersenne.ca it will recover all assignments for all systems/workers to the first worker. This will overwrite any existing work files.",
)
parser.add_option(
	"--register-exponents",
	action="store_true",
	help="Prompt for all parameters needed to register one or more specific exponents and exit.",
)
parser.add_option(
	"--unreserve",
	dest="exponent",
	type="int",
	help="Unreserve the exponent and exit. Use this only if you are sure you will not be finishing this exponent.",
)
parser.add_option(
	"--unreserve-all",
	action="store_true",
	dest="unreserve_all",
	help="Report assignment results, unreserve all assignments and exit.",
)
parser.add_option(
	"--no-more-work", action="store_true", dest="no_more_work", help="Prevent this program from getting new assignments and exit."
)
parser.add_option(
	"--resume-work",
	action="store_true",
	dest="resume_work",
	help="Resume getting new assignments after having previously run the --no-more-work option and exit.",
)
parser.add_option("--ping", action="store_true", dest="ping", help="Ping the PrimeNet server, show version information and exit.")
parser.add_option("--no-color", action="store_false", dest="color", help="Do not use color in output.")
parser.add_option("--setup", action="store_true", help="Prompt for all the options that are needed to setup this program and exit.")

# TODO: add detection for most parameter, including automatic change of the hardware
memory = get_physical_memory() or 1024
cores, threads = get_cpu_cores_threads()
cache_sizes = get_cpu_cache_sizes()

group = optparse.OptionGroup(
	parser,
	"Registering Options",
	"Sent to PrimeNet/GIMPS when registering. It will automatically send the progress, which allows the program to then be monitored on the GIMPS website CPUs page (https://www.mersenne.org/cpus/), just like with Prime95/MPrime. This also allows the program to get much smaller Category 0 and 1 exponents, if it meets the other requirements (https://www.mersenne.org/thresholds/).",
)
group.add_option(
	"-H", "--hostname", dest="computer_id", default=platform.node()[:20], help="Optional computer name, Default: %default"
)
group.add_option(
	"--cpu-model", dest="cpu_brand", default=get_cpu_model() or "cpu.unknown", help="Processor (CPU) model, Default: %default"
)
group.add_option("--features", dest="cpu_features", default="", help="CPU features, Default: '%default'")
group.add_option(
	"--frequency",
	dest="cpu_speed",
	type="int",
	default=get_cpu_frequency() or 1000,
	help="CPU frequency/speed (MHz), Default: %default MHz",
)
group.add_option(
	"--memory", dest="memory", type="int", default=memory, help="Total physical memory (RAM) (MiB), Default: %default MiB"
)
group.add_option(
	"--max-memory",
	dest="day_night_memory",
	type="int",
	default=int(0.9 * memory),
	help="Configured day/night P-1 stage 2 memory (MiB), Default: %default MiB (90% of physical memory). Required for P-1 assignments.",
)
group.add_option(
	"--max-disk-space",
	dest="worker_disk_space",
	type="float",
	default=0.0,
	help="Configured disk space limit per worker to store the proof interim residues files for PRP tests (GiB/worker), Default: %default GiB/worker. Use 0 to not send.",
)
group.add_option(
	"--l1",
	dest="cpu_l1_cache_size",
	type="int",
	default=cache_sizes[1] or 8,
	help="L1 Data Cache size (KiB), Default: %default KiB",
)
group.add_option(
	"--l2", dest="cpu_l2_cache_size", type="int", default=cache_sizes[2] or 512, help="L2 Cache size (KiB), Default: %default KiB"
)
group.add_option(
	"--l3", dest="cpu_l3_cache_size", type="int", default=cache_sizes[3], help="L3 Cache size (KiB), Default: %default KiB"
)
group.add_option(
	"--cores", dest="num_cores", type="int", default=cores or 1, help="Number of physical CPU cores, Default: %default"
)
group.add_option(
	"--hyperthreads",
	dest="cpu_hyperthreads",
	type="int",
	default=-(threads // -cores) if cores else 0,
	help="Number of CPU threads per core (0 is unknown), Default: %default. Choose 1 for non-hyperthreaded and 2 or more for hyperthreaded.",
)
group.add_option(
	"--hours",
	dest="cpu_hours",
	type="int",
	default=24,
	help="Hours per day you expect the GIMPS program will run (1 - 24), Default: %default hours. Used to give better estimated completion dates.",
)
parser.add_option_group(group)

group = optparse.OptionGroup(
	parser,
	"Notification Options",
	"Optionally configure this program to automatically send an e-mail/text message notification if there is an error, if the GIMPS program has stalled, if the available disk space is low or if it found a new Mersenne prime. Send text messages by using your mobile providers e-mail to SMS or MMS gateway. Use the --test-email option to verify the configuration.",
)
group.add_option(
	"--to",
	dest="toemails",
	action="append",
	default=[],
	help="To e-mail address. Use multiple times for multiple To/recipient e-mail addresses. Defaults to the --from value if not provided.",
)
group.add_option("-f", "--from", dest="fromemail", help="From e-mail address")
group.add_option(
	"-S",
	"--smtp",
	dest="smtp",
	help="SMTP server. Optionally include a port with the 'hostname:port' syntax. Defaults to port 465 with --tls and port 25 otherwise.",
)
group.add_option("--tls", action="store_true", dest="tls", help="Use a secure connection with SSL/TLS")
group.add_option("--starttls", action="store_true", dest="starttls", help="Upgrade to a secure connection with StartTLS")
group.add_option("-U", "--email-username", dest="email_username", help="SMTP server username")
group.add_option("-P", "--email-password", dest="email_password", help="SMTP server password")
group.add_option("--test-email", action="store_true", dest="test_email", help="Send a test e-mail message and exit")
parser.add_option_group(group)

opts_no_defaults = optparse.Values()
_, args = parser.parse_args(values=opts_no_defaults)
if args:
	parser.error("Unexpected argument: {0!r}".format(args[0]))
options = optparse.Values(parser.get_default_values().__dict__)
options._update_careful(opts_no_defaults.__dict__)

workdir = os.path.expanduser(options.workdir)
dirs = [os.path.join(workdir, adir) for adir in options.dirs] if options.dirs else [workdir]

for adir in dirs:
	if not os.path.isdir(adir):
		parser.error("Directory {0!r} does not exist".format(adir))

logger = logging.getLogger()
logger.setLevel(max(logging.INFO - options.debug * 10, 0))
console_handler = logging.StreamHandler()
console_handler.setFormatter(
	ColorFormatter(
		"%(filename)s: "
		+ ("%(funcName)s:\t" if options.debug > 1 else "")
		+ "[%(threadName)s%(worker)s %(asctime)s]  %(levelname)s: %(message)s"
	)
)
logger.addHandler(console_handler)
# logging.basicConfig(level=max(logging.INFO - options.debug * 10, 0), format="%(filename)s: " + ("%(funcName)s:\t" if options.debug > 1 else "") + "[%(threadName)s %(asctime)s]  %(levelname)s: %(message)s")

# If debug is requested

# https://stackoverflow.com/questions/10588644/how-can-i-see-the-entire-http-request-thats-being-sent-by-my-python-application
if options.debug > 1:
	try:
		# Python 3+
		from http.client import HTTPConnection
	except ImportError:
		from httplib import HTTPConnection
	HTTPConnection.debuglevel = 1

	# You must initialize logging, otherwise you'll not see debug output.
	requests_log = logging.getLogger("requests.packages.urllib3")
	requests_log.setLevel(logging.DEBUG)
	requests_log.propagate = True

# Adapted from: https://github.com/python/cpython/blob/main/Lib/_colorize.py
COLOR = True
if os.name == "nt":  # Windows
	handle = ctypes.windll.kernel32.GetStdHandle(wintypes.DWORD(-12))  # STD_ERROR_HANDLE
	mode = wintypes.DWORD()
	if not ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
		COLOR = False
		# raise ctypes.WinError()
	elif not mode.value & 0x0004:  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
		COLOR = False

if COLOR:
	if "NO_COLOR" in os.environ:
		COLOR = False
	if options.color is not None and not options.color:
		COLOR = False
	if "FORCE_COLOR" in os.environ:
		COLOR = True

if not opts_no_defaults.__dict__ and not os.path.exists(os.path.join(workdir, options.localfile)):
	logging.info(
		"No command line arguments provided or %r file found, now running --setup to configure the program.", options.localfile
	)
	options.setup = True

# load prime.ini and update options
config = config_read()
config_updated = merge_config_and_options(config, options)

logfile = os.path.join(workdir, options.logfile)
maxBytes = config.getint(SEC.PrimeNet, "MaxLogFileSize") if config.has_option(SEC.PrimeNet, "MaxLogFileSize") else 2 * 1024 * 1024
file_handler = logging.handlers.RotatingFileHandler(logfile, maxBytes=maxBytes, backupCount=5, encoding="utf-8")
file_handler.setFormatter(Formatter("[%(threadName)s%(worker)s %(asctime)s]  %(levelname)s: %(message)s"))
logger.addHandler(file_handler)

logging.info("AutoPrimeNet assignment handler version %s, Python %s", VERSION, platform.python_version())

if not hasattr(opts_no_defaults, "work_preference"):
	for i in range(options.num_workers):
		section = "Worker #{0}".format(i + 1) if options.num_workers > 1 else SEC.PrimeNet
		if config.has_option(section, "WorkPreference"):
			options.work_preference.append(config.get(section, "WorkPreference"))

if not options.work_preference:
	options.work_preference = [str(PRIMENET.WP_PRP_FIRST)]

if options.setup:
	test_email = setup()
	config_write(config)

if options.results_file is None:
	options.results_file = "results.json.txt" if options.mfaktc or options.mfakto else "results.txt"
	config.set(SEC.PrimeNet, "resultsfile", options.results_file)
if options.days_of_work is None:
	options.days_of_work = 1.0 if options.mfaktc or options.mfakto else 3.0
	config.set(SEC.PrimeNet, "DaysOfWork", str(options.days_of_work))
if not config.has_option(SEC.PrimeNet, "MaxExponents"):
	amax = (10000 if options.min_exp and options.min_exp >= MAX_PRIMENET_EXP else 1000) if options.mfaktc or options.mfakto else 15
	config.set(SEC.PrimeNet, "MaxExponents", str(amax))

# check options after merging so that if prime.ini file is changed by hand,
# values are also checked
RE = re.compile(r"[^A-Za-z0-9_@+,./:()<>=! -]")
if options.user_id is not None:
	if len(options.user_id) > 20:
		parser.error("User ID must be less than or equal to 20 characters")
	res = RE.search(options.user_id)
	if res:
		logging.warning("User ID has invalid character: %r", res.group())
if options.computer_id is not None:
	if len(options.computer_id) > 20:
		parser.error("Computer name must be less than or equal to 20 characters")
	res = RE.search(options.computer_id)
	if res:
		logging.warning("Computer name has invalid character: %r", res.group())
if not 8 <= len(options.cpu_brand) <= 64:
	parser.error("CPU model must be between 8 and 64 characters")
res = RE.search(options.cpu_brand)
if res:
	logging.warning("CPU model has invalid character: %r", res.group())
if options.cpu_features is not None:
	if len(options.cpu_features) > 64:
		parser.error("CPU features must be less than or equal to 64 characters")
	res = RE.search(options.cpu_features)
	if res:
		logging.warning("CPU features has invalid character: %r", res.group())

PROGRAM = PROGRAMS[5 if options.mfakto else 4 if options.mfaktc else 3 if options.cudalucas else 2 if options.gpuowl else 1]

# Convert mnemonic-form worktypes to corresponding numeric value, check
# worktype value vs supported ones:
worktypes = {
	"Pfactor": PRIMENET.WP_PFACTOR,
	"SmallestAvail": PRIMENET.WP_LL_FIRST,
	"DoubleCheck": PRIMENET.WP_LL_DBLCHK,
	"WorldRecord": PRIMENET.WP_LL_WORLD_RECORD,
	"100Mdigit": PRIMENET.WP_LL_100M,
	"SmallestAvailPRP": PRIMENET.WP_PRP_FIRST,
	"DoubleCheckPRP": PRIMENET.WP_PRP_DBLCHK,
	"WorldRecordPRP": PRIMENET.WP_PRP_WORLD_RECORD,
	"100MdigitPRP": PRIMENET.WP_PRP_100M,
}
# {"PRP": 150, "PM1": 4, "LL_DC": 101, "PRP_DC": 151, "PRP_WORLD_RECORD": 152, "PRP_100M": 153, "PRP_P1": 154}
# this and the above line of code enables us to use words or numbers on the cmdline
SUPPORTED = frozenset(
	[PRIMENET.WP_FACTOR, PRIMENET.WP_GPU_FACTOR]
	if options.mfaktc or options.mfakto
	else (
		[PRIMENET.WP_PFACTOR, PRIMENET.WP_LL_FIRST, PRIMENET.WP_LL_DBLCHK, PRIMENET.WP_LL_WORLD_RECORD, PRIMENET.WP_LL_100M]
		+ (
			[
				PRIMENET.WP_PRP_FIRST,
				PRIMENET.WP_PRP_DBLCHK,
				PRIMENET.WP_PRP_WORLD_RECORD,
				PRIMENET.WP_PRP_100M,
				PRIMENET.WP_PRP_NO_PMINUS1,
			]
			+ (
				[PRIMENET.WP_PRP_COFACTOR, PRIMENET.WP_PRP_COFACTOR_DBLCHK]
				if not options.gpuowl
				else [106, PRIMENET.WP_PRP_DC_PROOF]
			)
			if not options.cudalucas
			else []
		)
	)
)

for i, work_preference in enumerate(options.work_preference):
	if work_preference in worktypes:
		options.work_preference[i] = work_preference = worktypes[work_preference]
	if not work_preference.isdigit():
		parser.error("Unrecognized work preference = {0}".format(work_preference))
	if int(work_preference) not in SUPPORTED:
		parser.error("Unsupported work preference = {0} for {1}".format(work_preference, PROGRAM["name"]))

# Convert first time LL worktypes to PRP
CONVERT_DICT = {
	PRIMENET.WP_LL_FIRST: PRIMENET.WP_PRP_FIRST,
	PRIMENET.WP_LL_WORLD_RECORD: PRIMENET.WP_PRP_WORLD_RECORD,
	PRIMENET.WP_LL_100M: PRIMENET.WP_PRP_100M,
}

work_preference = []
for awork_preference in options.work_preference:
	awork_preference = int(awork_preference)
	if awork_preference in CONVERT_DICT:
		awork_preference = CONVERT_DICT[awork_preference]
	work_preference.append(awork_preference)

if len(options.work_preference) == 1 and options.num_workers > 1:
	options.work_preference *= options.num_workers
	work_preference *= options.num_workers

if len(options.work_preference) != options.num_workers:
	parser.error(
		"The number of work preferences ({0:n}) must be 1 or equal to the number of workers ({1:n})".format(
			len(options.work_preference), options.num_workers
		)
	)

# if guid already exist, recover it, this way, one can (re)register to change
# the CPU model (changing instance name can only be done in the website)
guid = get_guid(config)
if options.user_id is None:
	parser.error("Username must be given")

if options.password is not None:
	parser.error("The legacy manual testing feature was deprecated and has been removed from this program")

if options.dirs and len(options.dirs) != options.num_workers:
	parser.error(
		"The number of directories ({0:n}) must be equal to the number of workers ({1:n})".format(
			len(options.dirs), options.num_workers
		)
	)

if not options.dirs and options.cpu >= options.num_workers:
	parser.error(
		"CPU core or GPU number ({0:n}) must be less than the number of workers ({1:n})".format(options.cpu, options.num_workers)
	)

if options.cert_work and not options.gpuowl:
	parser.error("Proof certification work is currently only supported by PRPLL")

if not 1 <= options.cert_cpu_limit <= 100:
	parser.error("Proof certification work limit must be between 1 and 100%")

if options.prpll:
	parser.error("PRPLL is not PrimeNet server compatible and is thus not (yet) fully supported.")

if not (options.mlucas or options.cudalucas or options.gpuowl or options.mfaktc or options.mfakto):
	parser.error("Must select at least one GIMPS program to get assignments for")

if (
	(options.mlucas and options.cudalucas)
	or (options.mlucas and options.gpuowl)
	or (options.mlucas and options.mfaktc)
	or (options.mlucas and options.mfakto)
	or (options.cudalucas and options.gpuowl)
	or (options.cudalucas and options.mfaktc)
	or (options.cudalucas and options.mfakto)
	or (options.gpuowl and options.mfaktc)
	or (options.gpuowl and options.mfakto)
	or (options.mfaktc and options.mfakto)
):
	parser.error("This program can only be used with Mlucas or GpuOwl/PRPLL or CUDALucas or mfaktc or mfakto")

if options.day_night_memory > options.memory:
	parser.error(
		"Max memory ({0:n} MiB) must be less than or equal to the total physical memory ({1:n} MiB)".format(
			options.day_night_memory, options.memory
		)
	)

if options.min_exp and options.max_exp and options.min_exp < MAX_PRIMENET_EXP <= options.max_exp:
	parser.error(
		"Minimum exponent ({0}) and maximum exponent ({1}) must both be less than or greater than 1,000,000,000 (for TF1G)".format(
			options.min_exp, options.max_exp
		)
	)

if not 0 <= options.days_of_work <= 180:
	parser.error("Days of work must be less than or equal to 180 days")

if not 1 <= options.cpu_hours <= 24:
	parser.error("Hours per day must be between 1 and 24 hours")

if options.convert_ll_to_prp and options.convert_prp_to_ll:
	parser.error("Cannot convert LL assignments to PRP and PRP assignments to LL at the same time")

if not 1 <= options.hours_between_checkins <= 7 * 24:
	parser.error("Hours between checkins must be between 1 and 168 hours (7 days)")

if (
	options.toemails
	or options.fromemail
	or options.smtp
	or options.tls
	or options.starttls
	or options.email_username
	or options.email_password
) and not (options.fromemail and options.smtp):
	parser.error("Providing the E-mail options requires also setting the SMTP server and From e-mail address")

if options.fromemail and options.smtp:
	toemails = options.toemails
	fromemail = options.fromemail

	EMAILRE = re.compile(
		r'^(?=.{6,254}$)(?=.{1,64}@)(([^@"(),:;<>\[\\\].\s]|\\[^():;<>.])+|"([^"\\]|\\.)+")(\.(([^@"(),:;<>\[\\\].\s]|\\[^():;<>.])+|"([^"\\]|\\.)+"))*@((xn--)?[^\W_]([\w-]{0,61}[^\W_])?\.)+(xn--)?[^\W\d_]{2,63}$',
		re.U,
	)

	for i, toemail in enumerate(toemails):
		email = parseaddr(toemail)
		_, toaddress = email
		temp = toaddress or toemail
		if not EMAILRE.match(temp):
			parser.error("{0!r} is not a valid e-mail address.".format(temp))
		toemails[i] = email

	email = parseaddr(fromemail)
	_, fromaddress = email
	temp = fromaddress or fromemail
	if not EMAILRE.match(temp):
		parser.error("{0!r} is not a valid e-mail address.".format(temp))
	fromemail = email

	toemails = toemails or [fromemail]

if options.tls and options.starttls:
	parser.error("Cannot use both SSL/TLS and StartTLS.")

if options.num_workers > 1:
	for i in range(options.num_workers):
		section = "Worker #{0}".format(i + 1)
		if not config.has_section(section):
			# Create the section to avoid having to test for it later
			config.add_section(section)

for i, awork_preference in enumerate(options.work_preference):
	section = "Worker #{0}".format(i + 1) if options.num_workers > 1 else SEC.PrimeNet
	if not config.has_option(section, "WorkPreference") or config.get(section, "WorkPreference") != awork_preference:
		logging.debug("update %r section %r with WorkPreference=%s", options.localfile, section, awork_preference)
		config.set(section, "WorkPreference", awork_preference)
		config_updated = True

# write back prime.ini if necessary
if config_updated:
	logging.debug("write %r", options.localfile)
	config_write(config)

if options.setup:
	register_instance(guid)
	if options.fromemail and options.smtp and test_email:
		test_msg(guid)
	executable = os.path.basename(sys.executable) if sys.executable else "python3"
	print(
		"""
Setup of this instance of AutoPrimeNet is now complete.
Run the below command each time you want to start the program. If you have more than one worker, add the -D/--dir option for each worker directory.

	{0}{1}

Then, start {2} as normal. AutoPrimeNet will automatically get assignments, report assignment progress, report results and upload proof files.
Run --help for a full list of available options.
""".format(
			sys.argv[0]
			if is_pyinstaller()
			else "{0} -OO {1}".format(executable[:-4] if executable.endswith(".exe") else executable, sys.argv[0]),
			" --dir <directory>" * options.num_workers if options.num_workers != 1 else "",
			PROGRAM["name"],
		)
	)
	sys.exit(0)

if options.timeout > options.hours_between_checkins * 60 * 60:
	parser.error(
		"Timeout ({0:n} seconds) should be less than or equal to the hours between checkins ({1:n} hours)".format(
			options.timeout, options.hours_between_checkins
		)
	)

if 0 < options.timeout < 30 * 60:
	parser.error(
		"Timeout ({0:n} seconds) must be greater than or equal to {1:n} seconds (30 minutes)".format(options.timeout, 30 * 60)
	)

if options.status:
	output_status(dirs)
	sys.exit(0)

if options.proofs:
	for i, adir in enumerate(dirs):
		adapter = logging.LoggerAdapter(logger, {"cpu_num": i} if options.dirs else None)
		cpu_num = i if options.dirs else options.cpu
		tasks = list(read_workfile(adapter, adir))
		submit_work(dirs, adapter, adir, cpu_num, tasks)
		upload_proofs(adapter, adir, cpu_num)
	sys.exit(0)

if options.recover or options.recover_all:
	recover_assignments(dirs, recover_all=options.recover_all)
	sys.exit(0)

if options.register_exponents:
	register_exponents(dirs)
	sys.exit(0)

if options.exponent:
	unreserve(dirs, options.exponent)
	sys.exit(0)

if options.unreserve_all:
	unreserve_all(dirs)
	sys.exit(0)

if options.no_more_work:
	logging.info("Not requesting any more work")
	config.set(SEC.PrimeNet, "QuitGIMPS", str(1))
	config_write(config)
	sys.exit(0)

if options.resume_work:
	if config.has_option(SEC.PrimeNet, "QuitGIMPS"):
		logging.info("Resuming getting new work.")
		config.remove_option(SEC.PrimeNet, "QuitGIMPS")
		config_write(config)
	sys.exit(0)

if options.ping:
	result = ping_server()
	if result is None:
		logging.error("Failure pinging server")
		sys.exit(1)
	logging.info("\n%s", result)
	sys.exit(0)

if options.test_email:
	if not options.fromemail or not options.smtp:
		parser.error("The SMTP server and From e-mail address are required to send e-mails")
	if not test_msg(guid):
		sys.exit(1)
	sys.exit(0)

# use the v5 API for registration and program options
if guid is None:
	register_instance(guid)
	if options.timeout <= 0:
		sys.exit(0)
# worktype has changed, update worktype preference in program_options()
elif config_updated:
	register_instance(guid)

logging.info("Monitoring director%s: %s", "y" if len(dirs) == 1 else "ies", ", ".join(map(repr, map(os.path.abspath, dirs))))

while True:
	start = timeit.default_timer()
	config = config_read()
	current_time = time.time()
	last_time = config.getint(SEC.Internals, "LastEndDatesSent") if config.has_option(SEC.Internals, "LastEndDatesSent") else 0
	checkin = options.timeout <= 0 or current_time >= last_time + options.hours_between_checkins * 60 * 60
	last_time = last_time if checkin else None

	if config.has_option(SEC.PrimeNet, "CertGetFrequency"):
		cert_freq = config.getfloat(SEC.PrimeNet, "CertGetFrequency")
	elif options.cert_cpu_limit >= 50:
		cert_freq = 0.5
	else:
		cert_freq = (
			3
			if options.num_cores >= 20
			else 4
			if options.num_cores >= 12
			else 6
			if options.num_cores >= 7
			else 8
			if options.num_cores >= 3
			else 12
			if options.num_cores >= 2
			else 24
		)
	cert_freq = max(0.25, cert_freq)
	cert_last_update = (
		config.getint(SEC.Internals, "CertDailyRemainingLastUpdate")
		if config.has_option(SEC.Internals, "CertDailyRemainingLastUpdate")
		else 0
	)
	cert_work = current_time >= cert_last_update + cert_freq * 60 * 60

	check_disk_space(dirs)

	for i, adir in enumerate(dirs):
		adapter = logging.LoggerAdapter(logger, {"cpu_num": i} if options.dirs else None)
		cpu_num = i if options.dirs else options.cpu
		workfile = os.path.join(adir, options.worktodo_file)
		with LockFile(workfile):
			process_add_file(adapter, adir)
			tasks = deque(read_workfile(adapter, adir))  # list
			submit_work(dirs, adapter, adir, cpu_num, tasks)
			registered = register_assignments(adapter, adir, cpu_num, tasks)
			progress = update_progress_all(adapter, adir, cpu_num, last_time, tasks, checkin or registered)
			if cert_work:
				get_cert_work(adapter, adir, cpu_num, current_time, progress, tasks)
			get_assignments(adapter, adir, cpu_num, progress, tasks)

		download_certs(adapter, adir, tasks)

		if options.timeout <= 0:
			upload_proofs(adapter, adir, cpu_num)

	start_time = config.getint(SEC.Internals, "RollingStartTime") if config.has_option(SEC.Internals, "RollingStartTime") else 0
	if current_time >= start_time + 6 * 60 * 60:
		adjust_rolling_average(dirs)

	if cert_work:
		config.set(SEC.Internals, "CertDailyRemainingLastUpdate", str(int(current_time)))

	if checkin:
		config.set(SEC.Internals, "LastEndDatesSent", str(int(current_time)))
	config_write(config)
	if options.timeout <= 0:
		logging.info("Done communicating with server.")
		break
	logging.debug("Done communicating with server.")
	thread = threading.Thread(target=aupload_proofs, name="UploadProofs", args=(dirs,))
	thread.start()
	elapsed = timeit.default_timer() - start
	if options.timeout > elapsed:
		logging.info(
			"Will report results%s and upload proof files every %.01f hour%s, and report progress every %s hour%s. Next check at: %s",
			"" if config.has_option(SEC.PrimeNet, "QuitGIMPS") and config.getboolean(SEC.PrimeNet, "QuitGIMPS") else ", get work",
			options.timeout / (60 * 60),
			"s" if options.timeout != 60 * 60 else "",
			options.hours_between_checkins,
			"s" if options.hours_between_checkins != 1 else "",
			(datetime.fromtimestamp(current_time) + timedelta(seconds=options.timeout)).strftime("%c"),
		)
		try:
			time.sleep(options.timeout - elapsed)
		except KeyboardInterrupt:
			break
	thread.join()

sys.exit(0)
