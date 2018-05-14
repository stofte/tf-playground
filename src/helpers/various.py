import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltim
from IPython.display import display, HTML
from JSAnimation.IPython_display import display_animation
from matplotlib import animation

def output_folder(prefix='output-', folderpath=['logs']):
    relpath = os.path.join(os.path.dirname(__file__), '..', '..', *folderpath)
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    f = '{}{}'.format(prefix, ts)
    return os.path.abspath(os.path.join(relpath, f))

def folder_size(path='.'):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += folder_size(entry.path)
    return total

def strip_consts(tf, graph_def, max_const_size=32):
    """
    Strip large constant values from graph_def.
    """
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def tf_show_graph(tf, max_const_size=32):
    """
    Visualize TensorFlow graph.
    """
    graph_def = tf.get_default_graph().as_graph_def()
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(tf, graph_def, max_const_size=max_const_size)
    code = """
        <script>
            function load() {{
            document.getElementById("{id}").pbtxt = {data};
            }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
            <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:100%;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))    

def display_frames_as_video(frames):
    ax = plt.axes([0,0,1,1], frameon=False)
    img = plt.imshow(frames[0])


    def animate(i):
        img.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=25)
    display(HTML(anim.to_html5_video()))
    plt.close()
