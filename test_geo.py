# -*- coding: utf-8 -*-
# 2016/07/03 作成
# 国土地理院： http://vldb.gsi.go.jp/sokuchi/surveycalc/surveycalc/bl2stf.html

from geo import LatLng, Bowring, Hubeny, Angle, Length
import numpy as np
from time import time
from math import modf
import copy, matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------
# 関数定義
# ----------------
# 度分秒⇒10進数への変換
def dms2deg(dms) :
    dms_float = float(dms)
    dms_float, d = modf( dms_float*1e-4 )
    dms_float, m = modf( dms_float*1e2 )
    s = dms_float*1e2
    return d + m/60. + s/3600.


# --------------------------------------------------------
# テスト用計算
# 国土地理院で一括計算した結果ファイル (.out) を元に実行
# --------------------------------------------------------

# 結果格納用変数の初期化
x_lng1 = { '1': [], '2': [], '3a': [], '3b1': [], '3b2': [], '3b3': [] }
result = {
    "y_dist": copy.deepcopy( x_lng1 ),
    "y_azim": copy.deepcopy( x_lng1 ),
    "e_dist": copy.deepcopy( x_lng1 ),
    "e_azim": copy.deepcopy( x_lng1 ),
    "h_etime": copy.deepcopy( x_lng1 ),
}
resultset = { 
    "bowring": copy.deepcopy( result ),
    "hubeny": copy.deepcopy( result ),
}
formulas = {
    "bowring": Bowring,
    "hubeny": Hubeny,
}

# ファイルオープン
f = open('testdata_dms.out')
for i in range(6) :
    line = f.readline()

# ヘッダー出力
print u"schema  | lt1[deg]  lg1[deg] | lt2[deg]  lg2[deg] | zone  dist[km]  azim[deg]     time[sec]"
print u"---------------------------------------------------------------------------------------------------"

while line :
    # ファイル内容取得
    rows = line.split();
    lt1 = dms2deg(rows[0])
    lg1 = dms2deg(rows[1])
    lt2 = dms2deg(rows[2])
    lg2 = dms2deg(rows[3])
    dist_true = float(rows[4])
    azim_true = dms2deg(rows[5])

    latlng_orig = LatLng( Angle.from_deg(lt1), Angle.from_deg(lg1) )
    latlng_dest = LatLng( Angle.from_deg(lt2), Angle.from_deg(lg2) )

    for formula_name, Formula in formulas.iteritems():
        # 計算実行
        stime = time()
        formula = Formula(latlng_orig, latlng_dest)
        distance, azimuth = formula.calculate()
        etime = time() - stime
        if formula_name == 'bowring':
            zone = formula.common_params.zone

        # 結果格納
        dist = distance.to_kilometer()
        azim = azimuth.to_deg()
        resultset[formula_name]["y_dist"][zone].append(dist)
        resultset[formula_name]["y_azim"][zone].append(azim)
    
        e_azim = (azim-azim_true) %360.
        e_azim = e_azim if(e_azim<=180.) else 360. - e_azim
        resultset[formula_name]["e_azim"][zone].append( e_azim *1e3 )
        resultset[formula_name]["e_dist"][zone].append( abs(dist*1e3-dist_true) *1e3 )
        resultset[formula_name]["h_etime"][zone].append( etime *1e6 )

        # 計算結果出力（コンソール）        
        print "{formula_name:7s} | {lt1:8.2f}  {lg1:8.2f} | {lt2:8.2f}  {lg2:8.2f} | {zone:4s}  {dist:8.2f}  {azim:9.2f}  {etime:10E}".format(**locals())
        
    # 結果格納
    x_lng1[zone].append(lg1)
    
    # 次行読み込み
    line = f.readline()


# ゾーンとか色とか設定
zones = ["1", "2", "3a", "3b1", "3b2", "3b3"]
col = { "1":"b", "2":"g", "3a":"r", "3b1":"c", "3b2":"m", "3b3":"y",}

for formula_name, Formula in formulas.iteritems():
    # --------------------------
    # グラフ１
    # --------------------------
    fig = plt.figure( figsize=(10,8) )
    axs_dist, axs_azim = [], []
    axs_dist.append( plt.subplot2grid((2,2), (0,0)) )
    axs_dist.append( plt.subplot2grid((2,2), (0,1)) )
    axs_azim.append( plt.subplot2grid((2,2), (1,0)) )
    axs_azim.append( plt.subplot2grid((2,2), (1,1)) )
    
    for ax in axs_dist :
        for zone in zones :
            ax.plot( np.array(x_lng1[zone]), np.array(resultset[formula_name]["y_dist"][zone]), "o")
        ax.grid(True)
    
    for ax in axs_azim :
        for zone in zones :
            ax.plot( np.array(x_lng1[zone]), np.array(resultset[formula_name]["y_azim"][zone]), "o")
        ax.grid(True)
    
    
    # タイトル
    axs_dist[0].set_title('distance (whole)')
    axs_dist[1].set_title('distance (zoom)')
    axs_azim[0].set_title('azimuth (whole)')
    axs_azim[1].set_title('azimuth (zoom)')
    
    # ラベル
    axs_dist[0].set_ylabel('distance [km]')
    axs_azim[0].set_ylabel('azimuth [deg]')
    axs_azim[0].set_xlabel('orig longitude [deg]')
    axs_azim[1].set_xlabel('orig longitude [deg]')
    
    # 凡例
    plt.legend(zones, loc="best")
    
    # 範囲
    axs_dist[0].set_xlim(-180, +180)
    axs_azim[0].set_xlim(-180, +180)
    axs_dist[1].set_xlim(-1, +1)
    axs_azim[1].set_xlim(-1, +1)
    axs_dist[0].set_ylim(0, 25000)
    axs_dist[1].set_ylim(19900, 20050)
    axs_azim[0].set_ylim(0, 360)
    axs_azim[1].set_ylim(0, 360)
    
    # 保存
    plt.savefig("result_{0}.png".format(formula_name))


    # --------------------------
    # グラフ２
    # --------------------------
    fig = plt.figure( figsize=(10,8) )
    ax_etime = plt.subplot2grid((3,1), (0,0))
    ax_dist = plt.subplot2grid((3,1), (1,0))
    ax_azim = plt.subplot2grid((3,1), (2,0))
    
    cnt = 0
    for zone in zones :
        n = len(resultset[formula_name]["h_etime"][zone])
        x = np.arange(cnt, cnt + n)
        cnt += n
    
        ax_etime.bar( x, np.array(resultset[formula_name]["h_etime"][zone]), color=col[zone] )
        ax_dist.bar( x, np.array(resultset[formula_name]["e_dist"][zone]), color=col[zone] )
        ax_azim.bar( x, np.array(resultset[formula_name]["e_azim"][zone]), color=col[zone] )
    
    # グリッド
    ax_etime.grid(True)
    ax_dist.grid(True)
    ax_azim.grid(True)
    
    # タイトル
    ax_etime.set_title("elapsed time")
    ax_dist.set_title("estimated error (distance)")
    ax_azim.set_title("estimated error (azimuth)")
    
    # ラベル
    ax_etime.set_ylabel("etime [usec]")
    ax_dist.set_ylabel("error [mm]")
    ax_azim.set_ylabel("error [m deg]")
    ax_azim.set_xlabel("index")
    
    # 保存
    plt.savefig("bars_{0}.png".format(formula_name))
