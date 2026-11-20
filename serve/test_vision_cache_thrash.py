"""#21 (Strata-rdna3): once a conversation held more than 64 pictures returned by tools, every request encoded every one
of them again.  A request walks its pictures twice: `_note_unreadable_tool_images` encodes each tool picture to check
it reads, then `prepare` encodes them all through `Vision.encode_all`.  The first walk did not keep the request's own
pictures, so with 64 cached and one new picture it evicted the oldest, the second walk missed it and evicted the next,
and so on down the whole list.

    python -m serve.test_vision_cache_thrash
"""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from serve.frontend import ChatTemplate, images_of
from serve.server import ByteTokenizer, Service, Vision
from serve.test_vram import VramEngine

ROOT = Path(__file__).resolve().parents[1]


def counting_vision(d: Path) -> Vision:
    """The resident Vision (its own encode, cache and eviction) over a stand-in reader that counts encodes."""
    v = Vision.__new__(Vision)
    v.dir, v.cache, v.lock = d, {}, threading.Lock()
    v.load = lambda src: src.encode()
    v.normalize = staticmethod(lambda data: data)
    v.elastic, v.stopped = False, False
    v.encodes = 0

    class Pipe:
        last = ""

        def write(self, text):
            Pipe.last = text

        def flush(self):
            v.encodes += 1
            (d / Pipe.last.split()[2]).write_bytes(b"x")

        def readline(self):
            return "OK 3\n"

    pipe = Pipe()
    v.proc = SimpleNamespace(stdin=pipe, stdout=pipe, poll=lambda: None)
    return v


def conversation(n: int) -> list[dict]:
    """A user turn, then n tool results each holding one distinct picture (an agent reading a screenshot per step)."""
    msgs = [{"role": "user", "content": "go"}]
    for i in range(n):
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": [{"type": "image", "source": f"shot {i}"}]})
    return msgs


class ToolPictureCacheThrash(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.vision = counting_vision(self.dir)
        tok = ByteTokenizer()
        engine = VramEngine(tok)
        engine.loaded = True
        self.svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=self.vision)

    def request(self, n: int) -> int:
        """Both picture walks of one request (prepare's own order); returns how many pictures were encoded."""
        before = self.vision.encodes
        messages = conversation(n)
        self.svc._note_unreadable_tool_images(messages)
        self.vision.encode_all(images_of(messages))
        return self.vision.encodes - before

    def test_one_new_picture_past_the_cache_size_is_the_only_encode(self):
        self.assertEqual(self.request(64), 64)                 # cold: every picture once
        self.assertEqual(self.request(65), 1)                  # main before #21: 66 (every picture again, plus one)
        self.assertEqual(self.request(66), 1)                  # main before #21: 67, 68, ... per new picture
        self.assertTrue(all(p.exists() for p, _ in self.vision.cache.values()))   # every cached file is on disk
        self.assertEqual(self.request(66), 0)                  # the same request again (a client's retry): nothing

    def test_the_cache_may_exceed_64_only_by_the_requests_own_pictures(self):
        self.request(64)
        self.request(65)
        self.assertEqual(len(self.vision.cache), 65)           # all 65 held by the request that just ran
        self.assertEqual(len(list(self.dir.glob("*.sve"))), 65)
        other = [{"role": "user", "content": "hi"},
                 {"role": "tool", "tool_call_id": "x", "content": [{"type": "image", "source": "another conversation"}]}]
        self.svc._note_unreadable_tool_images(other)            # another conversation's picture: back to 64, oldest first
        self.assertEqual(len(self.vision.cache), 64)

    def test_user_pictures_never_thrashed(self):
        # the user's own pictures only go through encode_all: unchanged by #21, checked here for symmetry
        msgs = [{"role": "user", "content": [{"type": "image", "source": f"u {i}"} for i in range(65)]}]
        self.vision.encode_all(images_of(msgs))
        before = self.vision.encodes
        self.vision.encode_all(images_of(msgs))
        self.assertEqual(self.vision.encodes - before, 0)


if __name__ == "__main__":
    unittest.main()
