# -*- coding: utf-8 -*-
# 2016/07/03 作成
# 国土地理院： http://vldb.gsi.go.jp/sokuchi/surveycalc/surveycalc/bl2stf.html
# 師匠の散歩： http://tancro.e-central.tv/grandmaster/excel/geo-compare.html （同じく国土地理院を参考にVBAで実装されている）
# YOLP2点間距離API：　http://developer.yahoo.co.jp/webapi/map/openlocalplatform/v1/distance.html

from math import pi, sin, cos, tan, atan, atan2, asin, sqrt, radians, degrees
sec = lambda x : 1. / cos( x )


# =====================================
# 長さ、角度の表現クラス
# =====================================
class Length(object):
    __slots__ = ['len_m']

    def __init__(self, len_m):
        self.len_m = len_m

    @staticmethod
    def from_meter(len_m):
        return Length( len_m )

    @staticmethod
    def from_kilometer(len_km):
        return Length( len_km *1e3 )

    def to_meter(self):
        return self.len_m

    def to_kilometer(self):
        return self.len_m *1e-3

class Angle(object):
    __slots__ = ['angle_rad']

    def __init__(self, angle_rad):
        self.angle_rad = angle_rad

    @staticmethod
    def from_rad(angle_rad):
        return Angle( angle_rad )

    @staticmethod
    def from_deg(angle_deg):
        return Angle( radians(angle_deg) )

    def to_rad(self):
        return self.angle_rad

    def to_deg(self):
        return degrees(self.angle_rad)


# =====================================
# (経度, 緯度)の表現クラス
# =====================================
class LatLng(object):
    __slots__ = ['lat', 'lng']

    def __init__(self, lat, lng):
        if -pi/2 <= lat.to_rad() <= pi/2 :
            self.lat = lat
        else:
            raise ValueError('latitude must be in the range [-90, 90] degrees')

        if -pi <= lng.to_rad() <= pi :
            self.lng = Angle.from_rad( lng.to_rad() %(2.*pi) )
        else:
            raise ValueError('longitude must be in the range [-180, 180] degrees')


# =====================================
# 測地定数クラス
# =====================================
class GeoConstants(object):
    __slots__ = ['a', 'f', 'b', 'e2', 'ep']
    a = 6378137.
    f = 1. / 298.257222101
    b = a * ( 1.-f )
    e2 = f * (2.-f)
    ep = 1./ ( 1./ e2 -1. )


# =====================================
# 公式クラスの共通インターフェース
# =====================================
class Formula(object):
    __slots__ = ['orig', 'dest']

    def __init__(self, orig, dest):
        self.orig = orig
        self.dest = dest

    def calculate(self):
        raise NotImplementedError


# =====================================
# Bowring公式の算出パラメータ
# =====================================
# ゾーン共通
class CommonParams(object):
    __slots__ = [
        'l_', 'L', 'L_', 'phi1', 'phi2', 'u1', 'u2',
        'Sigma', 'Delta', 'Sigma_', 'Delta_',
        'xi', 'eta', 'xi_', 'eta_', 'x', 'y', 'zone',
    ]

    def __init__(self, orig, dest):
        c = GeoConstants

        lat_orig = orig.lat.to_rad()
        lng_orig = orig.lng.to_rad()
        lat_dest = dest.lat.to_rad()
        lng_dest = dest.lng.to_rad()

        l_ = ( lng_dest - lng_orig + pi ) % (2.*pi) - pi
        L = abs( l_ )
        L_ = pi - L

        if 0. <= l_ :
            phi1 = lat_orig
            phi2 = lat_dest
        else:
            phi1 = lat_dest
            phi2 = lat_orig

        u1 = atan( (1.-c.f) * tan(phi1) )
        u2 = atan( (1.-c.f) * tan(phi2) )

        Sigma = phi2 + phi1
        Delta = phi2 - phi1
        Sigma_ = u2 + u1
        Delta_ = u2 - u1

        xi = cos(Sigma_/2.)
        xi_ = sin(Sigma_/2.)
        eta = sin(Delta_/2.)
        eta_ = cos(Delta_/2.)

        x = sin(u1) * sin(u2)
        y = cos(u1) * cos(u2)
        self.x, self.y = x, y

        z = x + y * cos(L)
        bound = - cos( radians(3.) * cos(u1) )
        zone = '1' if(0.<=z) else '2' if(bound<=z) else '3'
        
        self.l_, self.L, self.L_ = l_, L, L_
        self.phi1, self.phi2, self.u1, self.u2 = phi1, phi2, u1, u2
        self.Sigma, self.Delta, self.Sigma_, self.Delta_ = Sigma, Delta, Sigma_, Delta_ 
        self.xi, self.eta, self.xi_, self.eta_ = xi, eta, xi_, eta_ 
        self.x, self.y, self.zone = x, y, zone

# ゾーン３固有
class Zone3Params(object):
    __slots__ = [ 'R', 'd1', 'd2', 'q', 'f1', 'gamma0' ]
    
    def __init__(self, common_params):
        c = GeoConstants
        p = common_params

        coef1_R = c.f*pi * cos(p.u1)**2
        coef2_R = 1.+c.f - 3./4.*c.f * sin(p.u1)**2.
        R = coef1_R * ( 1. - c.f/4.* sin(p.u1)**2. * coef2_R )

        d1 = p.L_ * cos(p.u1) - R
        d2 = abs(p.Sigma_) + R

        q = p.L_ / ( c.f*pi )
        f1 = c.f/4. *( 1.+c.f/2. )

        gamma0 = q * ( 1. + f1 - f1 * q**2. )
        
        self.R, self.d1, self.d2, self.q, self.f1, self.gamma0 = R, d1, d2, q, f1, gamma0


# =====================================
# 反復計算パラメータ
# =====================================
# 共通インターフェース
class Iterator(object):
    __slots__ = [ 'CONVERSION_CRITERIA', 'MAX_ITERATION', 'cnt', 'isConverged' ]
    CONVERSION_CRITERIA = 1e-15
    MAX_ITERATION = 10
    cnt = 0
    isConverged = False

    def update(self):
        raise NotImplementedError
    
    def checkConversion(self, diff):
        if diff < self.CONVERSION_CRITERIA :
            self.isConverged = True
        
        if self.cnt < self.MAX_ITERATION :
            self.cnt += 1
        else:
            raise ArithmeticError('update {0} times but not converged'.format(self.MAX_ITERATION) )
            
# Theta
class ThetaIter(Iterator):
    def __init__(self, theta0, common_params):
        super(ThetaIter, self).__slots__.extend([
            'common_params', 'theta', 'F',
            'J', 'K', 'sigma', 'gamma', 'Gamma', 'zeta', 'zeta_',
            'func_gh', 'func_F',
        ])
        self.theta = theta0
        self.common_params = common_params
        
        if common_params.zone == '1' :
            self.func_gh = lambda a, b, x, y : sqrt( (a*x)**2 + (b*y)**2 )
            self.func_F = lambda theta, L, E : theta - L - E
        else:
            self.func_gh = lambda a, b, x, y : sqrt( (a*y)**2 + (b*x)**2 )
            self.func_F = lambda theta, L, E : theta - (pi-L) + E
        
    def update(self):
        c = GeoConstants
        p = self.common_params
        theta = self.theta
        
        g = self.func_gh( p.eta, p.xi, cos(theta/2.), sin(theta/2.) )
        h = self.func_gh( p.eta_, p.xi_, cos(theta/2.), sin(theta/2.) )

        J = 2. * g * h
        K = h**2. - g**2.
        sigma = 2. * atan2( g, h )
        gamma = p.y * sin( theta ) / J

        Gamma = 1. - gamma**2.
        zeta = Gamma*K - 2.*p.x
        zeta_ = zeta + p.x

        D = c.f/4. * ( 1.+c.f - 3./4. * c.f*Gamma )
        coef1_E = ( 1. - D*Gamma ) * c.f * gamma
        coef2_E = K * ( 2. * zeta**2. - Gamma**2. )
        E = coef1_E * ( sigma + D*J * ( zeta + D*coef2_E ) )
        F = self.func_F(theta, p.L, E)

        coef1_G = c.f*gamma**2 * ( 1. - 2. * D*Gamma )
        coef2_G = c.f * zeta_ * ( sigma/J ) * ( 1. - D*Gamma + (c.f*gamma**2) /2. )
        coef3_G = c.f**2. * zeta * zeta_ /4.
        G = coef1_G + coef2_G + coef3_G
        
        theta -= F/(1.-G)
        self.checkConversion( abs(F) )
        
        self.theta, self.F, self.J, self.K = theta, F, J, K
        self.sigma, self.gamma, self.Gamma = sigma, gamma, Gamma
        self.zeta, self.zeta_ = zeta, zeta_

# Gamma
class GammaIter(Iterator):
    def __init__(self, zone3_params):
        super(GammaIter, self).__slots__.extend(['zone3_params', 'gamma', 'Gamma', 'D'])
        self.zone3_params = zone3_params
        self.gamma = zone3_params.gamma0

    def update(self):
        c = GeoConstants
        p = self.zone3_params
        gamma = self.gamma
        Gamma = 1. - gamma**2.
        D = c.f/4. * ( 1.+c.f - 3./4. * c.f*Gamma )
        gamma_new = p.q / ( 1. - D*Gamma )
        self.checkConversion( abs( gamma_new - gamma ) )
        self.gamma, self.Gamma, self.D = gamma_new, Gamma, D


# =============================
# Bowring's Fomula による計算
# =============================
class Bowring(Formula):
    __slots__ = ['common_params','zone3_params','theta_iter']

    # --------------------
    # コンストラクタ
    # --------------------
    def __init__(self, orig, dest):
        super(Bowring, self).__init__(orig, dest)
        self.common_params = CommonParams(orig, dest)


    # --------------------
    # 距離と方位角の計算
    # --------------------
    def calculate(self):
        try:
            theta0 = self.__calculate_theta0()
        except OutOfZoneException:
            distance, azimuth = self.__calculate_when_3b23()
            return distance, azimuth

        theta_iter = ThetaIter(theta0, self.common_params)
        while not theta_iter.isConverged:
            theta_iter.update()
        self.theta_iter = theta_iter

        distance = self.__calculate_distance()
        azimuth = self.__calculate_azimuth()
        return distance, azimuth

    # 3b2/3b3の場合
    def __calculate_when_3b23(self):
        p = self.common_params
        p3 = self.zone3_params
        
        if p3.d1 == 0. :
            p.zone = '3b2'
            distance, azimuth = self.__calculate_when_3b2()
            return distance, azimuth

        elif p3.d1 < 0. :
            p.zone = '3b3'
            distance, azimuth = self.__calculate_when_3b3()
            return distance, azimuth

    # 3b2の場合
    def __calculate_when_3b2(self):
        p = self.common_params

        alpha1 = pi/2.
        Gamma = sin(p.u1) ** 2.
        distance = self.__calculate_distance_3b23(Gamma)

        return distance, Angle.from_rad(alpha1)

    # 3b3の場合
    def __calculate_when_3b3(self):
        p = self.common_params
        p3 = self.zone3_params
        
        g = GammaIter(p3)
        while not g.isConverged:
            g.update()

        m = 1. - p3.q * sec(p.u1)
        n = g.D*g.Gamma / ( 1.-g.D*g.Gamma )
        w = m - n + m*n
        alpha1 = pi/2. if(w<=0.) else pi/2. - 2.*asin( sqrt(w/2.) )
        alpha1 = alpha1 if(0.<=p.l_) else 2.*pi - alpha1
        
        distance = self.__calculate_distance_when_3b23(g.Gamma)
        return distance, Angle.from_rad(alpha1)
        

    # --------------------
    # theta0の計算
    # --------------------
    def __calculate_theta0(self):
        p = self.common_params
        c = GeoConstants

        if p.zone == '1' :
            return p.L * ( 1. + c.f * p.y )
                
        elif p.zone == '2' :
            return p.L_
        
        elif p.zone == '3' :
            self.zone3_params = p3 = Zone3Params(p)

            if p.Sigma != 0. :
                p.zone = '3a'
                return self.__calculate_theta0_when_3a()
            
            elif p.Sigma == 0. :
                if 0. < p3.d1 :
                    p.zone = '3b1'
                    return p.L_
                    
                elif p3.d1 <= 0.:
                    raise OutOfZoneException('cannot init theta cuz zone is 3b2 or 3b3')

    # 3aの場合
    def __calculate_theta0_when_3a(self):
        p = self.common_params
        p3 = self.zone3_params
        c = GeoConstants
        
        A0 = atan2( p3.d1, p3.d2 )
        B0 = asin( p3.R / sqrt( p3.d1**2. + p3.d2**2. ) )
        psi = A0 + B0

        j = p3.gamma0 / cos(p.u1)
        k = ( 1. + p3.f1 ) * abs(p.Sigma_) * ( 1. - c.f*p.y ) / ( c.f*p.y * pi )

        j1 = j / ( 1. + k * sec(psi) )
        psi_ = asin(j1)
        psi__ = asin( j1 * cos(p.u1) / cos(p.u2) )
        
        tan_psi = tan( (psi_+psi__) /2. )
        sin_Sigma_ = sin( abs(p.Sigma_) /2. )
        cos_Delta_ = cos( p.Delta_ /2. )
        theta0 = 2. * atan2( tan_psi * sin_Sigma_, cos_Delta_ )
        
        return theta0


    # --------------------
    # 距離の計算
    # --------------------
    def __calculate_distance(self):
        c = GeoConstants
        th = self.theta_iter

        bunbo = ( sqrt(1. + th.Gamma*c.ep) + 1. ) ** 2.
        n0 = th.Gamma*c.ep / bunbo
        n02 = n0**2.
        A = ( 1. + n0 ) * ( 1. + 5./4. * n02 )
        B = c.ep * ( 1. - 3./8. * n02 ) / bunbo

        coef1_C = th.K * ( 2.*th.zeta**2. - th.Gamma**2. )
        coef2_C = ( 1. - 4.*th.K**2. ) * ( 3.*th.Gamma**2. - 4.*th.zeta**2. )
        C = th.sigma - B*th.J * ( th.zeta + B/4. * ( coef1_C + B*th.zeta/6. * coef2_C ) )
        dist_m = c.a * ( 1.-c.f ) * A * C
        return Length.from_meter(dist_m)
            
    # 3b2, 3b3の場合
    def __calculate_distance_when_3b23(self, Gamma):
        c = GeoConstants
        n0 = Gamma*c.ep / ( sqrt(1. + Gamma*c.ep) + 1. ) ** 2.
        A = ( 1. + n0 ) * ( 1. + 5./4. * n0**2. )
        dist_m = c.a * ( 1.-c.f ) * A * pi
        return Length.from_meter(dist_m)


    # --------------------
    # 方位角の計算
    # --------------------
    def __calculate_azimuth(self):
        p = self.common_params
        th = self.theta_iter

        if p.zone == '1' :
            alpha = atan2( p.xi * tan(th.theta/2.), p.eta )
            beta = atan2( p.xi_ * tan(th.theta/2.), p.eta_ )
            alpha2_func = lambda alpha_, beta : alpha_ + beta
        else :
            alpha = atan2( p.eta_ * tan(th.theta/2.), p.xi_ )
            beta = atan2( p.eta * tan(th.theta/2.), p.xi )
            alpha2_func = lambda alpha_, beta : pi - alpha_ - beta
            
        if ( 0.<=alpha and 0.<=p.L ) or ( alpha<0. and p.L==0. ) :
            alpha_ = alpha
        elif ( alpha<0. and 0.<p.L ) :
            alpha_ = alpha + pi
            
        if ( alpha==0. and p.L==0. and 0.<p.Delta ) or ( alpha==0. and abs(p.L)==pi and 0.<th.sigma ) :
            alpha_ = 0.
        elif ( alpha==0. and p.L==0. and p.Delta<0. ) or ( alpha==0. and abs(p.L)==pi and th.sigma<0. ) :
            alpha_ = pi
            
        alpha1_ = alpha_ - beta
        alpha2 = alpha2_func( alpha_, beta )
        alpha21_ = pi + alpha2
        alpha1 = alpha1_ if(0.<=p.l_) else alpha21_

        return Angle.from_rad(alpha1)


# ===========================
# Hubeny Formula による計算
# ===========================
class Hubeny(Formula):
    
    def calculate(self):
        c = GeoConstants
        
        lat_dest = self.dest.lat.to_rad()
        lng_dest = self.dest.lng.to_rad()
        lat_orig = self.orig.lat.to_rad()
        lng_orig = self.orig.lng.to_rad()
        
        d_lng = lng_dest - lng_orig
        d_lat = lat_dest - lat_orig
        mu_lat = ( lat_dest + lat_orig ) /2.
        
        W = sqrt( 1. - c.e2 * sin(mu_lat)**2. )
        M = c.a * (1. - c.e2) / W**3.
        N = c.a / W
        
        D_lat = d_lat * M
        D_lng = d_lng * N * cos(mu_lat)
        
        dist_m = sqrt( D_lng**2. + D_lat**2. )
        azim_rad = atan2( D_lat, D_lng ) + pi/2.
        azim_rad = azim_rad %(2.*pi)

        return Length.from_meter( dist_m ), Angle.from_rad( azim_rad )
        
        
# ===========================
# ユーザー定義Exception
# ===========================
class OutOfZoneException(Exception): pass
