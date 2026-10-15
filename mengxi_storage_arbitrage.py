# -*- coding: utf-8 -*-
"""
蒙西市场工商业储能峰谷套利模型（MILP）
========================================
落地案例：蒙西某高载能深加工产业园（年用电量约2.5亿kWh，两级制110kV用户）
电价依据（全部来自官方公开文件）：
  1. 内蒙古电力(集团)代理购电工商业用户电价表（2025年10月，110kV两部制）：
     代理购电价格 0.3286 元/kWh、线损 0.0092、输配电价 0.0525、系统运行费 0.0222、
     政府性基金及附加 0.022425 → 平段电度电价 0.434925 元/kWh
  2. 《关于完善蒙西电网工商业分时电价政策的通知(试行)》(内发改价费字〔2023〕1630号)：
     大风季(1-5月、9-12月) 峰平谷比价 1.68:1:0.48
       峰 06:00-08:00、18:00-22:00；平 04:00-06:00、08:00-11:00、16:00-18:00、22:00-24:00；
       谷 00:00-04:00、11:00-16:00（午间谷：蒙西新能源大发特征）
     小风季(6-8月) 峰平谷比价 1.54:1:0.44
       峰 06:00-08:00、18:00-22:00；谷 11:00-16:00；
       尖峰 19:00-21:00（峰基础上浮20%）；深谷 13:00-15:00（谷基础上下浮20%）
  3. 蒙西现货市场（全国第5个正式运行）：申报价上限 1.5 元/kWh、下限 -0.05 元/kWh
     （负电价允许出现，反映新能源高占比下的供需特征——见README扩展方向）

模型：24时段 MILP，两充两放套利调度 + 容量配置寻优 + 投资收益测算（回收期/IRR）

运行：python mengxi_storage_arbitrage.py    依赖：numpy, scipy
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# ======================================================================
# 1. 蒙西分时电价构建（110kV两部制，元/kWh）
# ======================================================================
AGENT_BUY = 0.3286                       # 代理购电价格（2025年10月）
FIXED_110KV = 0.0092 + 0.0525 + 0.0222 + 0.022425   # 线损+输配+系统运行+基金

def price_windy(agent=AGENT_BUY, fixed=FIXED_110KV):
    """大风季（1-5月、9-12月）24小时电价，比价1.68:1:0.48"""
    pk, ff, vy = agent * 1.68 + fixed, agent + fixed, agent * 0.48 + fixed
    types = (['valley'] * 4 + ['flat'] * 2 + ['peak'] * 2 + ['flat'] * 3 +
             ['valley'] * 5 + ['flat'] * 2 + ['peak'] * 4 + ['flat'] * 2)
    return np.array([{'peak': pk, 'flat': ff, 'valley': vy}[t] for t in types])

def price_summer(agent=0.3377, fixed=FIXED_110KV):
    """小风季（6-8月）24小时电价，比价1.54:1:0.44，含尖峰/深谷
    （agent取2025年6月代理购电价格0.3377元/kWh）"""
    pk, ff, vy = agent * 1.54 + fixed, agent + fixed, agent * 0.44 + fixed
    jp, sv = agent * 1.54 * 1.2 + fixed, agent * 0.44 * 0.8 + fixed
    types = (['flat'] * 6 + ['peak'] * 2 + ['flat'] * 3 +
             ['valley'] * 2 + ['deep_valley'] * 2 + ['valley'] * 1 +
             ['flat'] * 2 + ['peak'] * 1 + ['jp'] * 2 + ['peak'] * 1 + ['flat'] * 2)
    m = {'peak': pk, 'flat': ff, 'valley': vy, 'jp': jp, 'deep_valley': sv}
    return np.array([m[t] for t in types])

# ======================================================================
# 2. 园区负荷曲线（年用电2.5亿kWh的高载能深加工园区典型日）
# ======================================================================
def park_load():
    """连续生产型深加工园区：基础负荷+早/晚双峰，日均约66万kWh（年约2.4亿kWh）"""
    h = np.arange(24)
    load = 26.0 + 5.5 * np.exp(-((h - 7.5) / 2.2) ** 2) \
               + 4.5 * np.exp(-((h - 19.5) / 2.5) ** 2)   # MW
    return load

# ======================================================================
# 3. 单日MILP调度（两充两放套利）
# ======================================================================
def solve_day(price, load_mw, p_max_mw=10.0, cap_mwh=20.0,
              eta_chg=0.94, eta_dis=0.94, soc_min=0.10, soc_max=0.95,
              soc_init=0.5, terminal=True):
    """
    min Σ price[t]·(load[t] + p_ch[t] − p_dis[t])
    s.t. SOC动态、充放互斥(0-1变量+大M)、SOC上下限、日循环（日末SOC≥初始）
    返回 (日购电成本元, 调度明细dict)
    """
    # 单位统一
    T = 24
    load = load_mw * 1000.0                         # kW
    p_max = p_max_mw * 1000.0                       # kW
    cap = cap_mwh * 1000.0                          # kWh

    PCH = np.arange(T); PDIS = PCH + T
    ZC = PDIS + T; ZD = ZC + T                      # ZC 充电开关 、 ZD 放电开关
    E = ZD + T                                      # SOC状态
    n = 5 * T

    c = np.zeros(n)
    c[PCH] = price                                  # 充电=加价购电
    c[PDIS] = -price                                # 放电=减少购电

    lb = np.zeros(n); ub = np.full(n, np.inf)
    ub[PCH] = p_max; ub[PDIS] = p_max               #  充放电上限功率
    ub[ZC] = 1.0; ub[ZD] = 1.0                      #  充放电开关变量
    lb[E] = soc_min * cap; ub[E] = soc_max * cap    #  电池状态的上下限
    if terminal:                                    #  是否强制日循环
        lb[E[-1]] = max(lb[E[-1]], soc_init * cap)  # 日循环：不透支电池
    integ = np.zeros(n); integ[ZC] = 1; integ[ZD] = 1   #

    rows, rlb, rub = [], [], []
    def add(coef, lo, hi):
        rows.append(coef); rlb.append(lo); rub.append(hi)

    for t in range(T):
        # 大M充放互斥
        r = np.zeros(n); r[PCH[t]] = 1; r[ZC[t]] = -p_max; add(r, -np.inf, 0)
        r = np.zeros(n); r[PDIS[t]] = 1; r[ZD[t]] = -p_max; add(r, -np.inf, 0)
        r = np.zeros(n); r[ZC[t]] = 1; r[ZD[t]] = 1; add(r, -np.inf, 1)
        # 禁止向电网倒送：放电不得超过本时段负荷
        r = np.zeros(n); r[PDIS[t]] = 1; add(r, -np.inf, load[t])
        # SOC动态
        r = np.zeros(n); r[E[t]] = 1.0
        r[PCH[t]] -= eta_chg; r[PDIS[t]] += 1.0 / eta_dis
        if t == 0:                              # 电池初始与末尾状态（一天当中）
            rhs = soc_init * cap
        else:
            r[E[t - 1]] -= 1.0; rhs = 0.0
        add(r, rhs, rhs)

    res = milp(c=c, constraints=LinearConstraint(np.vstack(rows), rlb, rub),
               integrality=integ, bounds=Bounds(lb, ub))
    assert res.success, res.message
    x = res.x
    d = {'p_ch': x[PCH], 'p_dis': x[PDIS], 'e': x[E], 'obj': float(res.fun)}
    return d['obj'], d


def no_storage_cost(price, load_mw):
    return float(np.sum(price * load_mw * 1000.0))


# ======================================================================
# 4. 年度收益与投资测算
# ======================================================================
DAYS_WINDY = 273        # 1-5月(151) + 9-12月(122)
DAYS_SUMMER = 92        # 6-8月

def annual_economics(p_max_mw, cap_mwh, capex_per_wh=0.75, om_ratio=0.015,
                     fade=0.03, years=10):
    """返回 dict：年度节省、CAPEX、O&M、静态回收期、10年IRR
    注意：solve_day 的目标函数只含储能充放电增量成本 Σprice·(p_ch−p_dis)，
    配储能后的日成本 = 无储能成本 + obj，故日节省 = −obj（obj必为负）。
    """
    load = park_load()
    obj_w, _ = solve_day(price_windy(), load, p_max_mw, cap_mwh)
    obj_s, _ = solve_day(price_summer(), load, p_max_mw, cap_mwh)

    save_year0 = (-obj_w * DAYS_WINDY - obj_s * DAYS_SUMMER)          # 元/年
    capex = cap_mwh * 1e6 * capex_per_wh                     # 元 (0.75元/Wh)
    om = om_ratio * capex                                     # 元/年

    # 静态回收期
    payback = capex / (save_year0 - om) if save_year0 > om else np.inf
    # IRR（容量年衰减fade，现金流 = 节省*衰减 - O&M，期末无残值）
    cf = [-capex] + [save_year0 * (1 - fade) ** y - om for y in range(years)]
    irr = _irr(cf)
    return {'save': save_year0, 'capex': capex, 'om': om,
            'payback': payback, 'irr': irr,
            'daily_windy': -obj_w, 'daily_summer': -obj_s}

def _irr(cf, lo=-0.5, hi=2.0, tol=1e-7):
    def npv(r):
        return sum(v / (1 + r) ** i for i, v in enumerate(cf))
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return float('nan')
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            break
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid


# ======================================================================
# 5. 主流程
# ======================================================================
def main():
    pw, ps = price_windy(), price_summer()
    load = park_load()
    print('=' * 70)
    print('蒙西工商业储能峰谷套利模型（110kV两部制，代理购电2025年10月口径）')
    print('=' * 70)
    print(f'大风季电价: 峰{pw.max():.4f} / 平{np.median(pw):.4f} / 谷{pw.min():.4f} '
          f'元/kWh | 峰谷价差 {pw.max()-pw.min():.4f}')
    print(f'小风季电价: 尖峰{ps.max():.4f} / 峰{np.sort(ps)[-3]:.4f} / 平{np.median(ps):.4f} '
          f'/ 谷{np.sort(ps)[0]:.4f} 元/kWh')
    print(f'园区负荷: 峰值{load.max():.1f}MW / 均值{load.mean():.1f}MW / '
          f'日电量{load.sum():.0f}MWh（年约{load.sum()*365*1000/1e8:.2f}亿kWh）')

    # ---- 典型日调度明细（10MW/20MWh）----
    cost_w, d_w = solve_day(pw, load, 10, 20)
    cost_s, d_s = solve_day(ps, load, 10, 20)
    print('\n【典型日两充两放调度】（10MW/20MWh，大风季）')
    print(f'{"h":>2} {"电价":>6} {"负荷MW":>7} {"充MW":>6} {"放MW":>6} {"SOC":>5}')
    for t in range(24):
        print(f'{t:2d} {pw[t]:6.4f} {load[t]:7.1f} '
              f'{d_w["p_ch"][t]/1000:6.1f} {d_w["p_dis"][t]/1000:6.1f} '
              f'{d_w["e"][t]/20000:5.2f}')
    print(f'大风季日节省: {-cost_w/1e4:.2f} 万元 | 小风季日节省: {-cost_s/1e4:.2f} 万元'
          f'（两充两放，日等效循环≈2次）')

    # ---- 容量配置寻优 ----
    print('\n【容量配置寻优】（EPC按0.75元/Wh，O&M按CAPEX1.5%/年，容量年衰减3%）')
    print(f'{"配置(MW/MWh)":>12} {"年节省(万元)":>10} {"CAPEX(万元)":>10} '
          f'{"静态回收期(年)":>12} {"10年IRR":>8}')
    best = None
    for p, c_ in [(5, 10), (10, 20), (15, 30), (20, 40)]:
        r = annual_economics(p, c_)
        irr_txt = f'{r["irr"]*100:9.1f}%' if np.isfinite(r["irr"]) else '       -  '
        print(f'{p:>5}/{c_:>6} {r["save"]/1e4:12.1f} {r["capex"]/1e4:12.1f} '
              f'{r["payback"]:14.2f} {irr_txt}')
        if best is None or r['irr'] > best[1]['irr']:
            best = ((p, c_), r)
    print(f'\n按IRR最优推荐配置: {best[0][0]}MW/{best[0][1]}MWh'
          f'（10年IRR {best[1]["irr"]*100:.1f}%，静态回收期 {best[1]["payback"]:.1f} 年）')

    # ---- 敏感性：代理购电价格 ----
    print('\n【敏感性：代理购电价格（峰谷价差随之变化），10MW/20MWh】')
    for ag in [0.25, 0.30, 0.3286, 0.36, 0.40]:
        r = _sens_agent(ag)
        print(f'  代理购电{ag:.4f}元 → 年节省{r/1e4:.1f}万元')

def _sens_agent(ag):
    load = park_load()
    pw = price_windy(ag)
    ps = price_summer(ag)
    ow, _ = solve_day(pw, load, 10, 20)
    os_, _ = solve_day(ps, load, 10, 20)
    return (-ow * DAYS_WINDY - os_ * DAYS_SUMMER)


if __name__ == '__main__':
    main()
