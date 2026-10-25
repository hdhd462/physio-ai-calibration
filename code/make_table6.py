import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats
import os

OUT = DATA_DIR

# 加载两库
ptb = np.load(f'{OUT}\\ptbxl_100hz.npz', allow_pickle=True)
chp = np.load(f'{OUT}\\chapman_100hz.npz', allow_pickle=True)

print('=== 数据加载 ===')
print(f'PTB-XL: n={len(ptb["y"])}, AF={ptb["y"].sum()} ({ptb["y"].mean()*100:.1f}%)')
print(f'Chapman: n={len(chp["y"])}, AF={chp["y"].sum()} ({chp["y"].mean()*100:.1f}%)')

# PTB-XL sex: 0=Male, 1=Female (需确认编码方向)
# Chapman: 0=Male, 1=Female (prep脚本里已编码)
def baseline_row(name, y, age, sex):
    df = pd.DataFrame({'AF': y, 'age': age, 'sex': sex})
    af = df[df.AF == 1]
    non = df[df.AF == 0]
    # 年龄
    age_af, age_non = af.age.dropna(), non.age.dropna()
    if len(age_af) > 0 and len(age_non) > 0:
        t, p_age = stats.ttest_ind(age_af, age_non)
        age_line = f'{age_af.mean():.1f}±{age_af.std():.1f} vs {age_non.mean():.1f}±{age_non.std():.1f}'
    else:
        age_line = 'NA'
        p_age = np.nan
    # 性别（女性占比）
    f_af = (af.sex == 1).sum() / max(len(af), 1) * 100
    f_non = (non.sex == 1).sum() / max(len(non), 1) * 100
    if len(af) > 0 and len(non) > 0:
        contingency = [[(af.sex==1).sum(), (af.sex==0).sum()],
                       [(non.sex==1).sum(), (non.sex==0).sum()]]
        chi2, p_sex, dof, expected = stats.chi2_contingency(contingency)
    else:
        p_sex = np.nan
    return {
        '数据库': name,
        '总记录数': len(df),
        'AF阳性': int(y.sum()),
        'AF率(%)': round(y.mean()*100, 1),
        '年龄_AF': age_line.split(' vs ')[0],
        '年龄_非AF': age_line.split(' vs ')[1] if ' vs ' in age_line else 'NA',
        '女性%_AF': round(f_af, 1),
        '女性%_非AF': round(f_non, 1),
        'P_年龄': f'{p_age:.3f}' if not np.isnan(p_age) else 'NA',
        'P_性别': f'{p_sex:.3f}' if not np.isnan(p_sex) else 'NA',
    }

# PTB-XL sex编码确认
print(f'\nPTB-XL sex分布: {dict(zip(*np.unique(ptb["sex"], return_counts=True)))}')
print(f'Chapman sex分布: {dict(zip(*np.unique(chp["sex"], return_counts=True)))}')

# PTB-XL age有异常值(300)，过滤
ptb_age = ptb['age'].copy()
ptb_age[ptb_age > 120] = np.nan
print(f'PTB-XL age范围(过滤后): {np.nanmin(ptb_age)}-{np.nanmax(ptb_age)}, 缺失={np.isnan(ptb_age).sum()}')
print(f'Chapman age范围: {np.nanmin(chp["age"])}-{np.nanmax(chp["age"])}, 缺失={np.isnan(chp["age"]).sum()}')

# 生成Table 6
rows = [
    baseline_row('PTB-XL (德国)', ptb['y'], ptb_age, ptb['sex']),
    baseline_row('Chapman (中国)', chp['y'], chp['age'], chp['sex']),
]
tbl6 = pd.DataFrame(rows)
print('\n=== Table 6: 两库基线特征对比 ===')
print(tbl6.to_string(index=False))

# 保存
tbl6.to_csv(f'{OUT}\\table6_baseline.csv', index=False, encoding='utf-8-sig')
print(f'\n保存: {OUT}\\table6_baseline.csv')

# 亚组统计（外部验证用）
print('\n=== Chapman年龄/性别分层AF率 ===')
chp_df = pd.DataFrame({'AF': chp['y'], 'age': chp['age'], 'sex': chp['sex']})
chp_df['age_grp'] = pd.cut(chp_df.age, bins=[0, 40, 60, 120], labels=['<40', '40-60', '>60'])
for grp in chp_df.age_grp.unique():
    sub = chp_df[chp_df.age_grp == grp]
    if len(sub) > 0:
        print(f'  年龄{grp}: n={len(sub)}, AF率={sub.AF.mean()*100:.1f}%')
for s, label in [(0, '男'), (1, '女')]:
    sub = chp_df[chp_df.sex == s]
    if len(sub) > 0:
        print(f'  {label}: n={len(sub)}, AF率={sub.AF.mean()*100:.1f}%')
