#!/usr/bin/env python3
"""把 cases/oat15a_2/cgrid_v12.cgns (5 块结构化, 准 2D) 转成求解器的 mesh.txt。

CGNS 里 h5py 轴序是 (nk, nj, ni); nk=2 且两个 k 面都是 symmetry -> 取 k=1 层塌成 2D。
单元 i/j 边界面的 BC 由 ZoneBC 的 PointRange 给出, 其余 i/j 边即块间对接面,
靠节点坐标去重后自然变成内部面。
"""
import argparse
import os
import sys

import h5py
import numpy as np

FAM2TYPE = {'wall': 'WALL', 'farfield': 'FAR'}


def get(g, name):
    """CGNS 节点名常带前导空格"""
    for k in g.keys():
        if k.strip() == name:
            return g[k]
    return None


def cstr(node):
    a = np.asarray(node)
    return ''.join(chr(c) for c in a.ravel() if c != 0)


def load(path):
    f = h5py.File(path, 'r')
    base = f['Base']
    zones = [k for k in base.keys()
             if isinstance(base[k], h5py.Group) and get(base[k], 'ZoneType') is not None]
    blocks = []
    for zn in zones:
        z = base[zn]
        ni, nj, nk = np.asarray(get(z, 'data'))[0]
        gc = get(z, 'GridCoordinates')
        X = np.asarray(get(get(gc, 'CoordinateX'), 'data'))[0]
        Y = np.asarray(get(get(gc, 'CoordinateY'), 'data'))[0]
        bc = []
        zb = get(z, 'ZoneBC')
        if zb is not None:
            for nm in zb.keys():
                gg = zb[nm]
                prn = get(gg, 'PointRange')
                if prn is None:
                    continue
                pr = np.asarray(get(prn, 'data'))
                fam = cstr(get(get(gg, 'FamilyName'), 'data')) if get(gg, 'FamilyName') else ''
                i1, j1, k1 = pr[0]
                i2, j2, k2 = pr[1]
                bc.append((nm.strip(), fam, int(i1), int(j1), int(i2), int(j2)))
        blocks.append(dict(name=zn, ni=int(ni), nj=int(nj), nk=int(nk),
                           X=X, Y=Y, bc=bc))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cgns')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--tol', type=int, default=11, help='节点合并的坐标取整位数')
    ap.add_argument('--drop-apex', type=int, default=0,
                    help='尾缘尖角局部粗化: 丢掉 blk-1/2 的第 2..M+1 条 j 线 与 '
                         'blk-3/5 的第 2..M+1 条 i 线 (通过块界面它们是同一族线, 必须同步丢)')
    ap.add_argument('--drop-te-j', type=int, default=0,
                    help='仅粗化尾缘附近的流向间距: 丢掉 blk-3/4/5 靠近 j=nj 的 M 条 j 线。'
                         'blk-1/2 与壁面完全不动')
    args = ap.parse_args()

    blocks = load(args.cgns)
    M, MJ = args.drop_apex, args.drop_te_j
    for b in blocks:
        ki, kj = list(range(b['ni'])), list(range(b['nj']))
        if M > 0:
            drop = set(range(1, M + 1))        # 原 1-based 的 2..M+1  ->  0-based 1..M
            if b['name'] in ('blk-1', 'blk-2'):
                kj = [j for j in kj if j not in drop]
            elif b['name'] in ('blk-3', 'blk-5'):
                ki = [i for i in ki if i not in drop]
        if MJ > 0:
            # 注意 j 方向: blk-3 的尾缘在 j=nj, blk-4/5 的尾缘在 j=1。
            # blk-3.i=1 与 blk-4.i=1 是同一条线但 j 序号相反, 所以要镜像丢,
            # 否则界面节点不再重合 (blk-4.i=15 与 blk-5.i=1 同向, 一起处理)。
            nj = b['nj']
            if b['name'] == 'blk-3':
                keep = set(range(0, nj - 1 - MJ)) | {nj - 1}
            elif b['name'] in ('blk-4', 'blk-5'):
                keep = {0} | set(range(1 + MJ, nj))
            else:
                keep = None
            if keep is not None:
                kj = [j for j in kj if j in keep]
        b['ki'], b['kj'] = ki, kj
    print(f'块数 {len(blocks)}   drop_apex={M}  drop_te_j={MJ}')
    tot_slots = 0
    for b in blocks:
        n = len(b['ki']) * len(b['kj'])
        tot_slots += n
        print(f"  {b['name']:>6}: {b['ni']:>4} x {b['nj']:>4} -> "
              f"{len(b['ki']):>4} x {len(b['kj']):>4}   节点槽 {n:>6}")

    # ---- 节点: 坐标去重 ----
    key2id = {}
    node_xy = []
    bid = []                      # 每块 (nj, ni) 的全局节点编号
    nsave = 0
    for b in blocks:
        ki, kj = b['ki'], b['kj']
        ids = np.empty((len(kj), len(ki)), dtype=np.int64)
        for jj, j in enumerate(kj):
            for ii, i in enumerate(ki):
                k = (round(float(b['X'][j, i]), args.tol),
                     round(float(b['Y'][j, i]), args.tol))
                nid = key2id.get(k)
                if nid is None:
                    nid = len(node_xy)
                    key2id[k] = nid
                    node_xy.append(k)
                else:
                    nsave += 1
                ids[jj, ii] = nid
        bid.append(ids)
    nnodes = len(node_xy)
    print(f'节点槽 {tot_slots} -> 唯一节点 {nnodes} (合并 {nsave})')
    if nsave == 0:
        print('错误: 块间节点没有合并, 说明块不共享节点', file=sys.stderr)
        return 1

    # ---- 单元 ----
    # edge key -> [cell ids...] ; 同时记录 bc 名
    edge_cells = {}
    cell_nodes = []
    cell_block = []
    cell_edges = []          # 每个单元 4 条边, 按 A-B, B-C, C-D, D-A 顺序
    for bi, b in enumerate(blocks):
        ni, nj, ids = len(b['ki']), len(b['kj']), bid[bi]
        for j in range(nj - 1):
            for i in range(ni - 1):
                cid = len(cell_nodes)
                quad = [ids[j, i], ids[j, i + 1], ids[j + 1, i + 1], ids[j + 1, i]]
                if len(set(int(q) for q in quad)) != 4:
                    print(f"错误: {b['name']} 单元(j={j+1},i={i+1}) 有重复节点 {quad}",
                          file=sys.stderr)
                    return 1
                cell_nodes.append(quad)
                cell_block.append(b['name'])
                eks = []
                for t in range(4):
                    a, c = int(quad[t]), int(quad[(t + 1) % 4])
                    ek = (a, c) if a < c else (c, a)
                    eks.append(ek)
                    edge_cells.setdefault(ek, []).append(cid)
                cell_edges.append(eks)

    ncells = len(cell_nodes)
    print(f'单元数 {ncells}')

    # ---- 边界 BC 表: 边 -> 名称 ----
    edge_bc = {}

    def mark(blk, pairs, fam):
        t = FAM2TYPE.get(fam)
        if t is None:
            return 0
        cnt = 0
        for a, b in pairs:
            ek = (int(a), int(b)) if a < b else (int(b), int(a))
            if ek[0] == ek[1]:
                continue
            prev = edge_bc.get(ek)
            if prev is not None and prev != t:
                print(f'错误: 边 {ek} BC 冲突 {prev} vs {t}', file=sys.stderr)
                sys.exit(1)
            edge_bc[ek] = t
            cnt += 1
        return cnt

    for bi, b in enumerate(blocks):
        ids = bid[bi]
        jpos = {j: p for p, j in enumerate(b['kj'])}
        ipos = {i: p for p, i in enumerate(b['ki'])}
        for (nm, fam, i1, j1, i2, j2) in b['bc']:
            if i1 == i2 and j1 != j2:             # i = const 线, j 变
                if (i1 - 1) not in ipos:
                    print(f"  警告: {b['name']}/{nm} 的 i={i1} 线已被丢弃, 跳过",
                          file=sys.stderr)
                    continue
                ii = ipos[i1 - 1]
                js = [jpos[j] for j in range(j1 - 1, j2) if j in jpos]
                pairs = [(ids[js[t], ii], ids[js[t + 1], ii]) for t in range(len(js) - 1)]
            elif j1 == j2 and i1 != i2:           # j = const 线, i 变
                if (j1 - 1) not in jpos:
                    print(f"  警告: {b['name']}/{nm} 的 j={j1} 线已被丢弃, 跳过",
                          file=sys.stderr)
                    continue
                jj = jpos[j1 - 1]
                iis = [ipos[i] for i in range(i1 - 1, i2) if i in ipos]
                pairs = [(ids[jj, iis[t]], ids[jj, iis[t + 1]]) for t in range(len(iis) - 1)]
            else:
                continue                          # k 面或退化 -> 2D 里丢弃
            n = mark(b, pairs, fam)
            if n:
                print(f"  {b['name']:>6}/{nm:>7} fam={fam:>9} -> {FAM2TYPE[fam]} 边 {n}")

    # ---- 组装面 ----
    interior, wall, far = [], [], []
    unassigned = []
    for ek, cs in edge_cells.items():
        if len(cs) == 2:
            interior.append((ek, cs[0], cs[1]))
        elif len(cs) == 1:
            t = edge_bc.get(ek)
            if t == 'WALL':
                wall.append((ek, cs[0]))
            elif t == 'FAR':
                far.append((ek, cs[0]))
            else:
                unassigned.append((ek, cs[0], cell_block[cs[0]]))
        else:
            print(f'错误: 边 {ek} 被 {len(cs)} 个单元共用', file=sys.stderr)
            return 1
    if unassigned:
        print(f'错误: {len(unassigned)} 条边界边没有 BC, 前 5 条:', file=sys.stderr)
        for u in unassigned[:5]:
            x, y = node_xy[u[0][0]]
            print(f'  {u[2]} 节点 {u[0]} xy≈({x:.6g},{y:.6g})', file=sys.stderr)
        return 1

    nfaces = len(interior) + len(wall) + len(far)
    print(f'面: 内部 {len(interior)}  壁面 {len(wall)}  远场 {len(far)}  合计 {nfaces}')

    # ---- 写文件 ----
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.cgns)), 'mesh.txt')
    # 边 -> 面编号; 单元行必须按环序给出 4 个面, 否则求解器 findnode 推出的
    # 节点顺序不成环, volume()/center() 的三角形求和会算错
    e2f = {}
    idx = 1
    for ek, c1, c2 in interior:
        e2f[ek] = idx
        idx += 1
    for ek, c1 in wall:
        e2f[ek] = idx
        idx += 1
    for ek, c1 in far:
        e2f[ek] = idx
        idx += 1

    with open(out, 'w') as fh:
        fh.write(f'{nnodes} {nfaces} {ncells} 3\n')
        fh.write('interior = INTER\nfix = WALL\nfarfield = FAR\n')
        fh.write('(node)\n')
        for i, (x, y) in enumerate(node_xy, 1):
            fh.write(f'{i} {x:.16e} {y:.16e}\n')
        fh.write('(end)\n(edge)\ninterior\n')
        idx = 1
        for ek, c1, c2 in interior:
            fh.write(f'{idx} {ek[0] + 1} {ek[1] + 1} {c1 + 1} {c2 + 1}\n')
            idx += 1
        fh.write('(end)\nfix\n')
        for ek, c1 in wall:
            fh.write(f'{idx} {ek[0] + 1} {ek[1] + 1} {c1 + 1} 0\n')
            idx += 1
        fh.write('(end)\nfarfield\n')
        for ek, c1 in far:
            fh.write(f'{idx} {ek[0] + 1} {ek[1] + 1} {c1 + 1} 0\n')
            idx += 1
        fh.write('(end)\n(end)\n(cell)\n')
        for c in range(ncells):
            fs = [e2f[ek] for ek in cell_edges[c]]
            if len(set(fs)) != 4:
                print(f'错误: 单元 {c + 1} 的面不成环 {fs}', file=sys.stderr)
                return 1
            fh.write(f'{c + 1} {fs[0]} {fs[1]} {fs[2]} {fs[3]}\n')
    print(f'已写出 {out}')

    # ---- 自检 ----
    print('--- 自检 ---')
    xy = np.array(node_xy)
    dup = len(node_xy) - len(set(node_xy))
    print(f'重复节点 {dup} (应为 0)')
    nonconvex = 0
    for quad in cell_nodes:
        p = xy[quad]
        sgn = 0
        bad = False
        for t in range(4):
            a = p[t]
            vec = p[(t + 1) % 4] - a
            for u in range(4):
                if u in (t, (t + 1) % 4):
                    continue
                cr = vec[0] * (p[u, 1] - a[1]) - vec[1] * (p[u, 0] - a[0])
                if cr == 0 or (sgn != 0 and (cr > 0) != (sgn > 0)):
                    bad = True
                    break
                sgn = cr
            if bad:
                break
        if bad:
            nonconvex += 1
    print(f'非凸/退化单元 {nonconvex} (应为 0)')
    xx, yy = xy[:, 0], xy[:, 1]
    print(f'坐标范围 X[{xx.min():.6g},{xx.max():.6g}] Y[{yy.min():.6g},{yy.max():.6g}]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
