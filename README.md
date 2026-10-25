# OddOne

**Can one half of a webcam image learn to see the other half?**

Split the webcam picture down the middle. The **visible** half is the only
input. The **hidden** half is the teacher, shown to the learner only while it
trains: ground truth, live. After training, press **H**: the hidden half is
switched off and replaced by what the lens *thinks* is there.

It's the A–B–C idea, and a cousin of Varjoluotain and KulmanTakaa: the hidden
half's light bounces, shadows and colour casts reach the visible half, and the
lens has to learn to read them back. Unlike Varjoluotain, there is **no
occluder model**. The lens learns whatever this particular room passes.

## Run it

```bash
pip install -r requirements.txt     # numpy, opencv-python; torch optional (CNN lens)
python oddone.py                    # Windows: if the camera is slow to open, add --dshow
python oddone.py --hidden left      # hide the left half instead
python oddone.py --no-cnn           # linear lens only
```

| key | does |
|---|---|
| `space` | learning on / off (freeze the lens to test it) |
| `h` | hide the hidden half; show the prediction in its place |
| `m` | switch shown lens: linear ↔ CNN |
| `c` | count channels: how many independent things the visible half tells about the hidden one (needs ~300 learned frames) |
| `e` | try to lock auto-exposure (see "the exposure trap" below) |
| `r` | reset everything |
| `s` / `l` | save / load the linear lens |
| `q`, `Esc` | quit |

The bottom strip shows the real hidden half, the lens's version, the error
×3, the two scores and the channel bars.

## The two lenses

- **Linear lens.** The best linear map from visible to hidden (the Wiener /
  linear MMSE estimator), learned from running covariances with slow
  forgetting. Light transport is linear, so this is the physics-shaped lens:
  `visible change = A · hidden change`, and it learns the inverse directly from
  data. Runs on CPU, about 4 ms per frame.
- **CNN lens** (if PyTorch is installed). A small encoder-decoder trained
  online on the GPU, your living mirror again, with the hidden half as
  teacher. Stronger, but also much better at **guessing from habit**.

## The score is honest

Every frame is predicted **before** the lens learns from it, so the score is
always on a frame the lens has never seen. It's the R² of the **change** in
the hidden half:

- **0** means no better than "it looks like it usually does";
- **1** means the change is predicted perfectly.

A lens that has merely memorised your room's usual look scores 0. Only
predicting what *changes* counts.

## The maths, and what it allows

Light is linear: `Δvisible = A · Δhidden + noise`, where A is the room's light
transport matrix (every bounce and shadow from the hidden half to each visible
pixel). The singular values of A decide how much of the hidden side can come
through: **the number above the noise floor is how many "pixels" of the hidden
half this room can show you.**

`sim_occluder.py` computes that for a flatland corner, with and without a door
edge:

![singular values](occluder_singular_values.png)

| | usable patterns at noise 1e-2 | at 1e-3 |
|---|---:|---:|
| bare wall (smooth falloff) | **3** | 4 |
| with a door edge | **91** | 119 |

A smooth wall blurs, its singular values fall off exponentially, and you get
"brighter left, brighter right" and little else. An **edge** puts a step into
the transport, the singular values fall only like 1/k, and dozens of patterns
survive. That is why corner cameras and Varjoluotain use an occluder. Here
the occluder doesn't need to be modelled, but it does need to **exist**: a door
frame, a chair leg, your own body.

The `c` key measures the live counterpart: held-out canonical correlations
between the two halves. Each bar above 0.5 is one independent channel through
which the visible half carries information about the hidden half.

## Reading vs guessing: the tests that matter

Anything the lens shows beyond what light carries comes from *habit*. To know
which you're seeing:

1. **Keep the hidden half's action out of the visible half.** If your arm
   crosses the midline, the lens just continues your arm. That's the easy
   case. The real test is a person or object moving **only** in the hidden half.
2. **Freeze and test** (`space` off). The score keeps counting on fresh frames.
3. **Novel object.** Put something in the hidden half that never appeared in
   training (an odd colour, a lamp switched on). If the lens gets *its* colour
   or position, it is reading light. If it draws a familiar thing, it is
   guessing.
4. **Remove the occluder.** Train with a chair or door edge near the boundary,
   then move it. If the lens was reading shadows, the score and the channel
   count should fall.
5. **Linear vs CNN.** If the CNN wins only on familiar scenes and loses on
   novel ones, the extra was habit, not sight.

## The exposure trap

A webcam's **auto-exposure** reacts to the whole image. When something bright
enters the hidden half, the whole frame dims, including the visible half. The
lens will learn that, and it's real information, but it's the camera's
light meter leaking, not the room. Press `e` to try locking exposure; whether
it works depends on the camera and driver. If it doesn't, expect one strong
channel (the first bar) that is mostly exposure.

## A good first setup

Point the camera so the boundary runs along a door frame or a table edge.
Put a wall or floor with light falling on it in the visible half and the space
where things move in the hidden half. Let it learn for a few minutes while
things happen *only* on the hidden side: walk by, switch a lamp on and off,
move a bright object. Then freeze, press `c`, and press `h`.

## Files

- `oddone.py`: the app (linear lens, optional CNN lens, honest scoring, channel count)
- `sim_occluder.py`: the singular-value simulation and plot
- `tests/`: the linear lens reads a known linear leak (R² > 0.8), does **not**
  hallucinate from unrelated input (R² < 0.1), the channel count recovers a
  known 3-channel leak, and an edge passes over 10× more than a bare wall

## Ledger

| claim | status |
|---|---|
| Light transport is linear; the best linear lens can be learned from paired frames | Established (Wiener / MMSE estimation) |
| An edge in the transport makes far more of the hidden side recoverable | Shown in simulation (3 vs 91 usable patterns); matches corner-camera literature |
| The lens needs no occluder model, only an occluder | Follows from the above; to be confirmed on a real room |
| How much *your* room shows | Unknown until you run it |
