
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt;
import numpy as np
import scipy.misc
import sklearn.cluster
import matplotlib.colors
import sys
import argparse
# import imageio
# import matplotlib.pyplot as plt

def cluster_im(im,type_cluster, num_k):
    print im.shape
    arr = np.array(im,dtype=float)/255.
    # print np.min(arr),np.max(arr)
    if type_cluster[0]=='hsv':
        arr = matplotlib.colors.rgb_to_hsv(arr)
    
    if len(type_cluster)>1:
        arr_org = np.array(arr)
        arr = arr[:,:,type_cluster[1]]
        arr = arr[:,:,np.newaxis]

    print 'CLUSTERING IN TO %d CLUSTERS' % num_k
    kmeans = sklearn.cluster.KMeans(n_clusters=num_k)

    arr_shape = arr.shape
    arr = np.reshape(arr, (arr.shape[0]*arr.shape[1],arr.shape[2]))
    mean = np.mean(arr,0,keepdims = True)
    std = np.std(arr,0, keepdims = True)
    std[std==0]=1.
    arr = (arr-mean)/std
    
    arr_idx = kmeans.fit_predict(arr)

    for idx_curr in range(num_k):
        arr[arr_idx==idx_curr,:]=(kmeans.cluster_centers_[idx_curr]*std)+mean

    arr = np.reshape(arr,arr_shape)
    
    if len(type_cluster)>1:
        arr_org[:,:,type_cluster[1]]=arr[:,:,0]
        arr = arr_org

    if type_cluster[0]=='hsv':
        arr = matplotlib.colors.hsv_to_rgb(arr)

    print 'DONE CLUSTERING'
    
    return arr

def downscale(im,big_side,interp):

    big_idx = 0 if im.shape[0]>im.shape[1] else 1

    small_idx = int(not big_idx)
    small_side = int(im.shape[small_idx] * float(big_side)/im.shape[big_idx])

    new_size = [0,0]
    new_size[big_idx] = big_side
    new_size[small_idx] = small_side
    
    print 'SCALING TO %d x %d' % (new_size[1],new_size[0])
    im_new = scipy.misc.imresize(im, tuple(new_size),interp= interp)
    print 'DONE SCALING'
    return im_new

def magic(in_file, 
    type_cluster, 
    num_clusters,
    big_side,
    sq_per_in,
    lab_inc,
    grid_color,
    show_out,
    ):

    in_file_stripped = in_file[:in_file.rindex('.')]
    
    im = scipy.misc.imread(in_file,mode='RGB')
    if np.max(im.shape)>800:
        im = downscale(im,800,'bilinear')

    im_new = cluster_im(im,[type_cluster],num_clusters)
    im_rs = downscale(im_new, big_side,'nearest')
    
    # im_new = im

    plt.ion()
    plt.figure()
    # plt.subplot(1,len(type_clusters),idx_type_cluster)
    plt.imshow(im_new)
    plt.title(' '.join([str(val) for val in type_cluster]))
    out_file = '_'.join([str(val) for val in [in_file_stripped,type_cluster,num_clusters,'clustered']])+'.png'
    plt.savefig(out_file)
    print 'CLUSTERING OUTPUT SAVED TO %s'% out_file
    # if show_out:
    #   plt.show()
    # else:
    # plt.close()

    plt.figure(figsize = (im_rs.shape[1]//sq_per_in,im_rs.shape[0]//sq_per_in))
    plt.imshow(im_rs, interpolation = 'nearest')
    ax = plt.gca()
    x_t = list(np.arange(-0.5, im_rs.shape[1]+0.5, 1))
    y_t = list(np.arange(-0.5, im_rs.shape[0]+0.5, 1))
    x_t_lab = [str(idx) if not idx%lab_inc else '' for idx, val in enumerate(x_t)]
    y_t_lab = [str(idx) if not idx%lab_inc else '' for idx, val in enumerate(y_t)]
    y_t_lab = y_t_lab[::-1]
    ax.set_xticks(x_t)
    ax.set_yticks(y_t)
    ax.set_xticklabels(x_t_lab)
    ax.set_yticklabels(y_t_lab)
    plt.grid(color=grid_color,linestyle='solid')
    
    out_file = '_'.join([str(val) for val in [in_file_stripped,type_cluster,num_clusters,big_side,'result']])+'.png'
    plt.savefig(out_file)
    print 'OUTPUT SAVED TO %s'% out_file
    if show_out:
        plt.show()
        print 'PRESS ANY KEY TO EXIT'
        raw_input()
    else:
        plt.close()


def main(args):

    parser = argparse.ArgumentParser(description='Make cross stitch pattern from image.')
    parser.add_argument('in_file', metavar='in_file', type=str, help='image to cross stitchofy')
    parser.add_argument('--type_cluster', metavar='type_cluster',default = 'rgb', type=str, help='clustering space. rgb or hsv')
    parser.add_argument('--num_clusters', metavar='num_clusters',default = 8, type=int, help='number of colors')
    parser.add_argument('--big_side', metavar='big_side',default = 80, type=int, help='number of boxes on the bigger side')

    parser.add_argument('--grid_color', metavar='grid_color', type=str, default='k', help='color of grid lines. pick high contrast for ease')
    parser.add_argument('--show_out', dest='show_out', default=False, action = 'store_true', help='set true to show images alongside saving them')
    
    
    parser.add_argument('--sq_per_in', metavar='sq_per_in',default = 5, type=int, help='squares per inch to display')
    parser.add_argument('--lab_inc', metavar='lab_inc',default = 5, type=int, help='incremenet between labels on both axes')

    # args = parser.parse_args(args[2:])
    args = vars(parser.parse_args(args[1:]))
    magic(**args)



if __name__=='__main__':
    main(sys.argv)

