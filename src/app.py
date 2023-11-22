import sys
import threading
import streamlit as st
from huggingface_hub import HfFolder, snapshot_download

@st.cache_data
def load_support():
    sys.path.append(snapshot_download("OpenShape/openshape-demo-support"))


# st.set_page_config(layout='wide')
load_support()


import numpy
import torch
import openshape
import transformers
from PIL import Image

@st.cache_resource
def load_openshape(name, to_cpu=False):
    pce = openshape.load_pc_encoder(name)
    if to_cpu:
        pce = pce.cpu()
    return pce


@st.cache_resource
def load_openclip():
    sys.clip_move_lock = threading.Lock()
    clip_model, clip_prep = transformers.CLIPModel.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        low_cpu_mem_usage=True, torch_dtype=half,
        offload_state_dict=True
    ), transformers.CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    if torch.cuda.is_available():
        with sys.clip_move_lock:
            clip_model.cuda()
    return clip_model, clip_prep


f32 = numpy.float32
half = torch.float16 if torch.cuda.is_available() else torch.bfloat16
# clip_model, clip_prep = None, None
clip_model, clip_prep = load_openclip()
model_b32 = load_openshape('openshape-pointbert-vitb32-rgb', True)
model_l14 = load_openshape('openshape-pointbert-vitl14-rgb')
model_g14 = load_openshape('openshape-pointbert-vitg14-rgb')
torch.set_grad_enabled(False)
for kc, vc in st.session_state.get('state_queue', []):
    st.session_state[kc] = vc
st.session_state.state_queue = []


import samples_index
from openshape.demo import misc_utils, classification, caption, sd_pc2img, retrieval


st.title("Multimodal Feature Alignment for 3D Representation Learning")
prog = st.progress(0.0, "Idle")
tab_cls, tab_text = st.tabs([
    "Classification",
    "Retrieval Text",
])


def sq(kc, vc):
    st.session_state.state_queue.append((kc, vc))


def reset_3d_shape_input(key):
    # this is not working due to streamlit problems, don't use it
    model_key = key + "_model"
    npy_key = key + "_npy"
    swap_key = key + "_swap"
    sq(model_key, None)
    sq(npy_key, None)
    sq(swap_key, "Y is up (for most Objaverse shapes)")


def auto_submit(key):
    if st.session_state.get(key):
        st.session_state[key] = False
        return True
    return False


def queue_auto_submit(key):
    st.session_state[key] = True
    st.experimental_rerun()


img_example_counter = 0


def image_examples(samples, ncols, return_key=None, example_text="Examples"):
    global img_example_counter
    trigger = False
    with st.expander(example_text, True):
        for i in range(len(samples) // ncols):
            cols = st.columns(ncols)
            for j in range(ncols):
                idx = i * ncols + j
                if idx >= len(samples):
                    continue
                entry = samples[idx]
                with cols[j]:
                    st.image(entry['dispi'])
                    img_example_counter += 1
                    with st.columns(5)[2]:
                        this_trigger = st.button('\+', key='imgexuse%d' % img_example_counter)
                    trigger = trigger or this_trigger
                    if this_trigger:
                        if return_key is None:
                            for k, v in entry.items():
                                if not k.startswith('disp'):
                                    sq(k, v)
                        else:
                            trigger = entry[return_key]
    return trigger


def demo_classification():
    with st.form("clsform"):
        load_data = misc_utils.input_3d_shape('cls')
        lvis_run = st.form_submit_button("Run Classification on LVIS Categories")
        if lvis_run or auto_submit("clsauto"):
            pc = load_data(prog)
            col2 = misc_utils.render_pc(pc)
            prog.progress(0.5, "Running Classification")
            pred = classification.pred_lvis_sims(model_g14, pc)
            with col2:
                for i, (cat, sim) in zip(range(5), pred.items()):
                    st.text(cat)
                    st.caption("Similarity %.4f" % sim)
            prog.progress(1.0, "Idle")


def demo_captioning():
    with st.form("capform"):
        load_data = misc_utils.input_3d_shape('cap')
        cond_scale = st.slider('Conditioning Scale', 0.0, 4.0, 2.0, 0.1, key='capcondscl')
        if st.form_submit_button("Generate a Caption") or auto_submit("capauto"):
            pc = load_data(prog)
            col2 = misc_utils.render_pc(pc)
            prog.progress(0.5, "Running Generation")
            cap = caption.pc_caption(model_b32, pc, cond_scale)
            st.text(cap)
            prog.progress(1.0, "Idle")
    if image_examples(samples_index.cap, 3, example_text="Examples (Choose one of the following 3D shapes)"):
        queue_auto_submit("capauto")


def retrieval_results(results):
    st.caption("Click the link to view the 3D shape")
    for i in range(len(results) // 4):
        cols = st.columns(4)
        for j in range(4):
            idx = i * 4 + j
            if idx >= len(results):
                continue
            entry = results[idx]
            with cols[j]:
                ext_link = f"https://objaverse.allenai.org/explore/?query={entry['u']}"
                st.image(entry['img'])
                # st.markdown(f"[![thumbnail {entry['desc'].replace('\n', ' ')}]({entry['img']})]({ext_link})")
                # st.text(entry['name'])
                quote_name = entry['name'].replace('[', '\\[').replace(']', '\\]').replace('\n', ' ')
                st.markdown(f"[{quote_name}]({ext_link})")


def retrieval_filter_expand(key):
    with st.expander("Filters"):
        sim_th = st.slider("Similarity Threshold", 0.05, 0.5, 0.1, key=key + 'rtsimth')
        tag = st.text_input("Has Tag", "", key=key + 'rthastag')
        col1, col2 = st.columns(2)
        face_min = int(col1.text_input("Face Count Min", "0", key=key + 'rtfcmin'))
        face_max = int(col2.text_input("Face Count Max", "34985808", key=key + 'rtfcmax'))
        col1, col2 = st.columns(2)
        anim_min = int(col1.text_input("Animation Count Min", "0", key=key + 'rtacmin'))
        anim_max = int(col2.text_input("Animation Count Max", "563", key=key + 'rtacmax'))
        tag_n = not bool(tag.strip())
        anim_n = not (anim_min > 0 or anim_max < 563)
        face_n = not (face_min > 0 or face_max < 34985808)
        filter_fn = lambda x: (
            (anim_n or anim_min <= x['anims'] <= anim_max)
            and (face_n or face_min <= x['faces'] <= face_max)
            and (tag_n or tag in x['tags'])
        )
        return sim_th, filter_fn


def demo_retrieval():
    with tab_text:
        with st.form("rtextform"):
            k = st.slider("Shapes to Retrieve", 1, 100, 16, key='rtext')
            text = st.text_input("Input Text", key="inputrtext")
            sim_th, filter_fn = retrieval_filter_expand('text')
            if st.form_submit_button("Run with Text") or auto_submit("rtextauto"):
                prog.progress(0.49, "Computing Embeddings")
                device = clip_model.device
                tn = clip_prep(
                    text=[text], return_tensors='pt', truncation=True, max_length=76
                ).to(device)
                enc = clip_model.get_text_features(**tn).float().cpu()
                prog.progress(0.7, "Running Retrieval")
                retrieval_results(retrieval.retrieve(enc, k, sim_th, filter_fn))
                prog.progress(1.0, "Idle")


try:
    with tab_cls:
        demo_classification()
    demo_retrieval()
except Exception:
    import traceback
    st.error(traceback.format_exc().replace("\n", "  \n"))
