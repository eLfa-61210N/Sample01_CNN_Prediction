

import numpy as np


# スカラーから1ofK配列を生成する
def one_hot( number, size=10):
    
    out_list = []
    
    for index in range(size):
        if index == number:
            out_list.append(1)
        else:
            out_list.append(0)

    return out_list


# テンソルを整形するための関数
def tensor_transform( input_tensor, filter_size, strides):

    output_tensor = np.array([])
    ts = input_tensor

    size_filter_y = filter_size[0]
    fy = size_filter_y
    size_filter_x = filter_size[1]
    fx = size_filter_x
    strides_y = strides[0]
    strides_x = strides[1]
    quantity_point_y = ( ( ts.shape[0] -fy) // strides_y) +1
    pqy = quantity_point_y
    quantity_point_x = ( ( ts.shape[1] -fx) // strides_x) +1
    pqx = quantity_point_x


    """
        各フィルタ左上を基準にポイント(インデクスとして)を設置
        そのポイントを左上としてフィルタを開く
    """
    filt_points = np.array([ ( strides_y * y, strides_x * x) for y in range( pqy) for x in range( pqx) ] )
    index_data = []


    for points in filt_points:

        p = points + [( y, x) for y in range( fy) for x in range( fx) ]
        index_data.append( p)


    for filt_num in range(len(index_data)): # インデクスのY軸個分のフィルタを生成する

        cache = []
        """
            先ずindex_dataのY軸を直接取り出し
                ・index_data の dimY は(index_X, index_Y)のように入っている
            index_X,Y からテンソルのXY座標を特定し
            仮バッファに貯める
            index_dataのY軸毎に出力に連結していく
        """
        for index_dim_y in index_data[filt_num]:
            cache.append( ts[ index_dim_y[0]][ index_dim_y[1]])

        output_tensor = np.append( output_tensor, cache)


    # ( dimY ポイントの個数, dimX 各カーネルの大きさ)で整形
    output_tensor = output_tensor.reshape( pqy * pqx, fy * fx)

    return output_tensor



