import tensorflow as tf
import tensorflow_datasets as tfds
from collections import defaultdict
import heapq as heap
from datasets import Dataset
import argparse

def coords_to_num(pair,dim=28):
    (vert,horiz)=pair
    return (dim*vert) + horiz

def num_to_coords(num,dim=28):
    vert=num//dim
    horiz=num%dim
    return (vert,horiz)

def valid_point(y,x,dim=28):
    if y<0 or y>dim-1 or x <0 or x>dim-1:
        return False
    return True

def find_start_points(img):
    nodes={}
    for y in range(len(img)):
        for x in range(len(img[y])):
            if img[y][x]>0:
                neighbors=0
                for d_y in [-1,0,1]:
                    for d_x in [-1,0,1]:
                        if valid_point(y+d_y, x+d_x) and img[y+d_y][x+d_x]>0:
                            neighbors+=1
                nodes[coords_to_num((y,x))]=neighbors
    least=min([v for v in nodes.values()])
    sorted_pairs= sorted([(k,v) for (k,v )in nodes.items() if v==least],key= lambda x: x[1])
    return [n for (n,e) in sorted_pairs]

def build_graph(img):
    dim=len(img)
    graph={}
    for y in range(dim):
        for x in range(dim):
            if img[y][x]<1:
                continue
            src=coords_to_num((y,x))
            #print(src,img[y][x] )
            if src not in graph:
                graph[src]=set()
            for d_y in [-1,0,1]:
                    for d_x in [-1,0,1]:
                        if valid_point(y+d_y, x+d_x) and  img[y+d_y][x+d_x]>0 and (d_y!=0 or d_x!=0):
                            dest=coords_to_num((y+d_y, x+d_x))
                            if dest not in graph:
                                graph[dest]=set()
                            graph[dest].add(src)
                            graph[src].add(dest)
    return graph

def dijkstra(graph, start_node):
    visited = set()
    parentsMap = {}
    pq = []
    nodeCosts = defaultdict(lambda: float('inf'))
    nodeCosts[start_node] = 0
    heap.heappush(pq, (0, start_node))

    while pq:
        # go greedily by always extending the shorter cost nodes first
        cost, node = heap.heappop(pq)
        visited.add(node)

        for adjNode in graph[node]:
            if adjNode in visited:	continue
                
            newCost = nodeCosts[node] + 1
            if nodeCosts[adjNode] > newCost:
                parentsMap[adjNode] = node
                nodeCosts[adjNode] = newCost
                heap.heappush(pq, (newCost, adjNode))
        
    return parentsMap, nodeCosts

def img_to_list(img):
    raw_img=tf.math.ceil(tf.squeeze(img)/255).numpy()
    sp=find_start_points(raw_img)
    graph=build_graph(raw_img)
    (parentsMap, nodeCosts)=dijkstra(graph,sp[0])
    return sorted([k for k in nodeCosts.keys()])

def make_dataset(limit,threshold=20):
    counts={x:0 for x in range(10)}
    data_list=[]
    (ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,)
    c=0
    for data in ds_train:
        c+=1
        if c >limit:
            break
        (img,label)=data
        digit=label.numpy()
        counts[digit]+=1
        sequence=img_to_list(img)
        if len(sequence)>= threshold:
            item={
                "label":digit,
                "sequence":sequence,
                "occurence":counts[digit],
                "split":"train"
            }
            data_list.append(item)
    c=0
    for data in ds_test:
        c+=1
        if c >limit:
            break
        (img,label)=data
        digit=label.numpy()
        counts[digit]+=1
        sequence=img_to_list(img)
        if len(sequence)>= threshold:
            item={
                "label":digit,
                "sequence":sequence,
                "occurence":counts[digit],
                "split":"test"
            }
            data_list.append(item)
    dataset = Dataset.from_list(data_list)
    return dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",type=int,default=150)
    parser.add_argument("--threshold",type=int,default=10)
    parser.add_argument("--dataset",type=str,default="mnist_dijkstra",help='name of datset to push to hub')
    parser.add_argument("--version",type=str,default="0.0")
    args = parser.parse_args()
    dataset=make_dataset(args.limit, threshold=args.threshold)
    #dataset.push_to_hub("jlbaker361/{}_v{}".format(args.dataset,args.version))