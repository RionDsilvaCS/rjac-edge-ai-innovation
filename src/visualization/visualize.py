from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

event_file = '/home/rion/light_proj/gits/edge_ai_seg_model/experiments/exp01/logs/exp_01_gscnn/version_0/events.out.tfevents.1724494049.VictusBaby.707.0'
save_pth = '/home/rion/light_proj/gits/edge_ai_seg_model/experiments/exp01/results/ver_0'

def save_graph(tag):
    x, y = [], []
    i = 1
    scalar_events = event_acc.Scalars(tag)
    for scalar_event in scalar_events:
        x.append(i)
        y.append(scalar_event.value)
        i += 1

    plt.ylim(0, 0.4)
    plt.plot(x, y)

    tag = tag.split('/')
    file_name = tag[0] + '.png'
    save_id = os.path.join(save_pth, file_name)
    plt.savefig(save_id)

    print("-:-:- {} loss graph done -:-:-".format(tag[0]))

    plt.clf()

    
event_acc = EventAccumulator(event_file)
event_acc.Reload() 

tags = event_acc.Tags()

for tag in tags['scalars']:
    save_graph(tag)
