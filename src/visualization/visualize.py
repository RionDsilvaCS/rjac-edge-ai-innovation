from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

event_file = r"E:\edge AI\rjac-edge-ai-innovation\experiments\exp02\logs\fcn_exp_02\version_18\events.out.tfevents.1724645022.SaiDheerajArikala.16592.0"
save_pth = r'E:\edge AI\rjac-edge-ai-innovation\experiments\exp02\Results'

def save_graph(tag):
    x, y = [], []
    i = 1
    scalar_events = event_acc.Scalars(tag)
    for scalar_event in scalar_events:
        x.append(i)
        y.append(scalar_event.value)
        i += 1

    plt.ylim(0, 1)
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
