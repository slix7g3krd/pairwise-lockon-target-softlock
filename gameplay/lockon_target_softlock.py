"""锁敌目标与镜头软锁（参考实现）。

本模块把题目中描述的六个缺陷各自收敛到一个可独立调用的入口：

1. ``los_every_frame`` —— 视线检查每帧重做，不复用按下锁定键那一帧的结果；
2. ``handle_generation`` —— 锁定句柄只保存目标 id + 代际，销毁后旧句柄不再持有
   已释放的变换引用；
3. ``stable_tiebreak`` —— 候选目标先按屏幕距离、距离相同再按稳定 id 排序，
   同一场景多次运行结果一致；
4. ``reset_interp`` —— 锁定切换时重置镜头插值参数与累计角，避免连续切换后
   镜头绕行整整一圈；
5. ``query_seq`` —— 异步视线射线带单调序号与目标代际，后返回的旧结果不会覆盖
   新目标的结果；
6. ``fallback_freecam`` —— 锁定丢失（视线中断、目标销毁、代际失配）时释放句柄并
   回退自由视角。

所有几何计算均为纯确定性计算，不使用随机数，也不依赖帧内缓存的遍历顺序。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]

#: 单次锁定切换允许的镜头朝向累计扫过角上限（度）。
MAX_SWITCH_SWEEP_DEG = 180.0

#: 镜头插值的默认平滑速率。
DEFAULT_INTERP_RATE = 12.0


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _segment_sphere_hit(p0: Vec3, p1: Vec3, center: Vec3, radius: float) -> bool:
    """线段 p0->p1 是否与球体相交（用于视线遮挡判定）。"""
    d = _sub(p1, p0)
    f = _sub(p0, center)
    a = _dot(d, d)
    if a <= 1e-12:
        return _dot(f, f) <= radius * radius
    b = 2.0 * _dot(f, d)
    c = _dot(f, f) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    root = math.sqrt(disc)
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 < t2)


def _wrap_deg(deg: float) -> float:
    """把角度归一化到 (-180, 180]，保证镜头永远走最短弧。"""
    wrapped = (deg + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        wrapped = 180.0
    return wrapped


def _camera_angles(camera: Vec3, target: Vec3) -> Tuple[float, float]:
    """返回相机看向目标所需的偏航、俯仰角（度）。"""
    dx = target[0] - camera[0]
    dy = target[1] - camera[1]
    dz = target[2] - camera[2]
    yaw = math.degrees(math.atan2(dx, -dz))
    horizontal = math.hypot(dx, dz)
    pitch = math.degrees(math.atan2(dy, horizontal))
    pitch = max(-89.0, min(89.0, pitch))
    return yaw, pitch


# ---------------------------------------------------------------------------
# 实体
# ---------------------------------------------------------------------------

@dataclass
class Target:
    """一个可锁定目标。

    ``generation`` 在目标重建 / 销毁时递增；``alive`` 为 False 时
    :meth:`transform` 返回 None，调用方不得再读取其位置。
    """

    tid: int
    pos: Vec3 = (0.0, 0.0, 0.0)
    screen: Vec2 = (0.0, 0.0)
    generation: int = 1
    alive: bool = True

    def screen_distance(self) -> float:
        """屏幕空间距原点的距离，用于候选排序。"""
        return math.hypot(self.screen[0], self.screen[1])

    def transform(self) -> Optional[Vec3]:
        """只在目标存活时返回位置，避免软锁读到已释放节点的零点。"""
        return self.pos if self.alive else None

    def destroy(self) -> "Target":
        self.alive = False
        self.generation += 1
        return self


@dataclass
class Occluder:
    """球形遮挡物。"""

    center: Vec3
    radius: float


@dataclass(frozen=True)
class LockHandle:
    """锁定句柄：只保存目标 id 与代际，绝不保存目标变换引用。"""

    tid: int
    generation: int


@dataclass
class LosQuery:
    """一次异步视线射线查询。"""

    seq: int
    tid: int
    generation: int
    frame: int
    resolved: bool = False
    visible: Optional[bool] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "tid": self.tid,
            "generation": self.generation,
            "frame": self.frame,
            "resolved": self.resolved,
            "visible": self.visible,
        }


@dataclass
class CameraInterp:
    """镜头插值状态。每次锁定切换 ``epoch`` 递增并清零累计角。"""

    epoch: int = 0
    t: float = 0.0
    rate: float = DEFAULT_INTERP_RATE
    yaw: float = 0.0
    pitch: float = 0.0
    yaw_accum: float = 0.0


# ---------------------------------------------------------------------------
# 世界状态与核心逻辑
# ---------------------------------------------------------------------------

def _initial_camera() -> Dict[str, Any]:
    return {"mode": "freecam", "locked": False, "yaw": 0.0, "pitch": 0.0}


@dataclass
class LockState:
    """锁定系统状态。可用于构造脚本化场景。"""

    camera_pos: Vec3 = (0.0, 0.0, 0.0)
    targets: Dict[int, Target] = field(default_factory=dict)
    occluders: List[Occluder] = field(default_factory=list)
    handle: Optional[LockHandle] = None
    frame: int = 0
    next_seq: int = 1

    queries: Dict[int, LosQuery] = field(default_factory=dict)
    applied_seq: Dict[int, int] = field(default_factory=dict)

    interp: CameraInterp = field(default_factory=CameraInterp)
    last_camera: Dict[str, Any] = field(default_factory=_initial_camera)
    freecam: bool = True

    # 计数器 / 审计数据
    los_checks: int = 0
    los_failures: int = 0
    los_log: List[Tuple[int, int, int, bool]] = field(default_factory=list)
    acquires: int = 0
    releases: int = 0
    switches: int = 0
    interp_resets: int = 0
    fallback_count: int = 0
    stale_results_dropped: int = 0
    query_issues: int = 0

    # -- 目标访问 ---------------------------------------------------------

    def live_candidates(self) -> List[Target]:
        return [t for t in self.targets.values() if t.alive]

    def get_target(self, tid: Optional[int]) -> Optional[Target]:
        if tid is None:
            return None
        return self.targets.get(tid)

    # -- 视线 -------------------------------------------------------------

    def check_los(self, tid: int) -> bool:
        """按当前几何重新计算视线；不做任何跨帧缓存。"""
        self.los_checks += 1
        target = self.targets.get(tid)
        if target is None or not target.alive:
            self.los_failures += 1
            self.los_log.append((self.frame, tid, -1, False))
            return False

        blocked = any(
            _segment_sphere_hit(self.camera_pos, target.pos, occ.center, occ.radius)
            for occ in self.occluders
        )
        visible = not blocked
        if not visible:
            self.los_failures += 1
        self.los_log.append((self.frame, tid, target.generation, visible))
        if len(self.los_log) > 4096:
            del self.los_log[:2048]
        return visible

    # -- 异步查询 ---------------------------------------------------------

    def issue_query(self, tid: int) -> LosQuery:
        target = self.targets[tid]
        query = LosQuery(
            seq=self.next_seq,
            tid=tid,
            generation=target.generation,
            frame=self.frame,
        )
        self.next_seq += 1
        self.queries[query.seq] = query
        self.query_issues += 1
        return query

    def resolve_query(self, seq: int, visible: bool) -> Dict[str, Any]:
        """按“新序号优先”合并异步结果。

        即使旧查询更晚完成，只要它的序号小于该目标已应用的最新序号，
        或它记录的目标代际已被重建，就会被丢弃。
        """
        query = self.queries.get(seq)
        if query is None:
            self.stale_results_dropped += 1
            return {"applied": False, "reason": "unknown_query", "seq": seq}

        target = self.targets.get(query.tid)
        if target is None or not target.alive or target.generation != query.generation:
            self.stale_results_dropped += 1
            query.resolved = True
            query.visible = None
            return {
                "applied": False,
                "reason": "stale_target",
                "seq": seq,
                "tid": query.tid,
            }

        latest = self.applied_seq.get(query.tid, 0)
        if seq < latest:
            self.stale_results_dropped += 1
            return {
                "applied": False,
                "reason": "stale_sequence",
                "seq": seq,
                "tid": query.tid,
                "latest_seq": latest,
            }

        self.applied_seq[query.tid] = seq
        query.resolved = True
        query.visible = bool(visible)
        return {
            "applied": True,
            "reason": "ok",
            "seq": seq,
            "tid": query.tid,
            "visible": query.visible,
        }

    def latest_async_visible(self, tid: int) -> Optional[bool]:
        seq = self.applied_seq.get(tid)
        if seq is None:
            return None
        query = self.queries.get(seq)
        return query.visible if query is not None else None


# ---------------------------------------------------------------------------
# 默认世界
# ---------------------------------------------------------------------------

def _default_state(seed: int = 0) -> LockState:
    """两个屏幕距离完全相同、按不同插入顺序放置的目标。

    稳定排序会始终先选 id 更小的目标，因此与字典/帧内缓存顺序无关。
    """
    state = LockState()
    first = Target(3, pos=(0.0, 0.0, -10.0), screen=(120.0, 80.0))
    second = Target(7, pos=(0.0, 0.0, -10.0), screen=(120.0, 80.0))
    if seed % 2:
        state.targets = {7: second, 3: first}
    else:
        state.targets = {3: first, 7: second}
    return state


def _as_state(value: Any) -> LockState:
    if isinstance(value, LockState):
        return value
    seed = value if isinstance(value, int) else 0
    return _default_state(seed)


def _coerce_handle(value: Any) -> Optional[LockHandle]:
    if isinstance(value, LockHandle):
        return value
    if isinstance(value, dict) and "tid" in value:
        return LockHandle(int(value["tid"]), int(value.get("generation", -1)))
    if isinstance(value, tuple) and len(value) >= 2:
        try:
            return LockHandle(int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _descriptor_parts(item: Any) -> Optional[Tuple[int, Vec2]]:
    """把 ``(tid, screen)`` / ``[tid, screen]`` 形式的描述子拆开。"""
    if isinstance(item, (tuple, list)) and len(item) == 2:
        tid_part, screen_part = item
        if isinstance(screen_part, (tuple, list)) and len(screen_part) == 2:
            try:
                return (
                    int(tid_part),
                    (float(screen_part[0]), float(screen_part[1])),
                )
            except (TypeError, ValueError):
                return None
    return None


def _is_candidate_collection(value: Any) -> bool:
    """判断一个位置参数是不是“候选目标集合”而不是世界状态。"""
    return isinstance(value, (list, tuple, set)) and not isinstance(value, str)


def _split_state_candidates(
    state: Any, candidates: Optional[Iterable[Any]]
) -> Tuple[LockState, Optional[Iterable[Any]]]:
    """允许 ``stable_tiebreak(candidates)`` 这类省略 state 的调用方式。"""
    if _is_candidate_collection(state):
        return _default_state(0), candidates if candidates is not None else state
    return _as_state(state), candidates


# ---------------------------------------------------------------------------
# 六个修复点 + 系统入口
# ---------------------------------------------------------------------------

def los_every_frame(state: Optional[LockState] = None) -> Dict[str, Any]:
    """推进一帧并重新计算当前锁定目标（或最优候选）的视线。

    每次调用都使用目标在当前帧的位置与当前的遮挡物重新求交，不读取
    按下锁定键那一帧缓存的结果。
    """
    s = _as_state(state)
    s.frame += 1

    if s.handle is not None:
        # 锁定中：先做代际校验，再逐帧算视线；任一失败当帧掉锁回自由视角。
        check = handle_generation(s, s.handle)
        if not check["valid"]:
            snapshot = fallback_freecam(s, check["reason"])
            snapshot.update(
                {
                    "frame": s.frame,
                    "tid": check["tid"],
                    "visible": False,
                    "checked": True,
                    "fresh": True,
                    "released": True,
                    "los_checks": s.los_checks,
                }
            )
            return snapshot
        tid: Optional[int] = s.handle.tid
    else:
        order = stable_tiebreak(s)
        tid = order[0] if order else None

    if tid is None:
        return {
            "frame": s.frame,
            "tid": None,
            "visible": False,
            "checked": False,
            "fresh": False,
            "released": False,
            "los_checks": s.los_checks,
        }

    visible = s.check_los(tid)
    if s.handle is not None and not visible:
        snapshot = fallback_freecam(s, "los_lost")
        snapshot.update(
            {
                "frame": s.frame,
                "tid": tid,
                "visible": False,
                "checked": True,
                "fresh": True,
                "released": True,
                "los_checks": s.los_checks,
            }
        )
        return snapshot

    return {
        "frame": s.frame,
        "tid": tid,
        "visible": visible,
        "checked": True,
        "fresh": True,
        "released": False,
        "los_checks": s.los_checks,
    }


def _is_handle_like(value: Any) -> bool:
    if isinstance(value, LockHandle):
        return True
    if isinstance(value, dict):
        return "tid" in value
    if isinstance(value, tuple) and len(value) == 2:
        return all(isinstance(part, int) for part in value)
    return False


def handle_generation(
    state: Optional[LockState] = None,
    handle: Optional[LockHandle] = None,
) -> Dict[str, Any]:
    """校验锁定句柄的目标代际；销毁/重建后的旧句柄一律失效。"""
    if handle is None and _is_handle_like(state):
        # 允许 handle_generation(handle) 这种只传句柄的调用。
        handle, state = state, None
    s = _as_state(state)
    raw = handle if handle is not None else s.handle
    h = _coerce_handle(raw)
    if h is None:
        return {
            "valid": False,
            "reason": "no_handle",
            "tid": None,
            "generation": None,
            "current_generation": None,
            "alive": False,
        }

    target = s.targets.get(h.tid)
    if target is None:
        return {
            "valid": False,
            "reason": "missing_target",
            "tid": h.tid,
            "generation": h.generation,
            "current_generation": None,
            "alive": False,
        }
    if not target.alive:
        return {
            "valid": False,
            "reason": "target_destroyed",
            "tid": h.tid,
            "generation": h.generation,
            "current_generation": target.generation,
            "alive": False,
        }
    if target.generation != h.generation:
        return {
            "valid": False,
            "reason": "stale_generation",
            "tid": h.tid,
            "generation": h.generation,
            "current_generation": target.generation,
            "alive": True,
        }
    return {
        "valid": True,
        "reason": "ok",
        "tid": h.tid,
        "generation": h.generation,
        "current_generation": target.generation,
        "alive": True,
    }


def stable_tiebreak(
    state: Optional[LockState] = None,
    candidates: Optional[Iterable[Any]] = None,
) -> List[int]:
    """按 (屏幕距离, 稳定目标 id) 排序候选目标，返回 id 列表。

    距离相同的时候不再依赖帧内缓存的遍历顺序，同一场景任意次运行结果一致。
    """
    s, candidates = _split_state_candidates(state, candidates)
    if candidates is None:
        items: List[Any] = list(s.targets.values())
    else:
        items = list(candidates)

    keys: List[Tuple[float, int]] = []
    for item in items:
        screen: Optional[Vec2] = None
        if isinstance(item, Target):
            target: Optional[Target] = item
        elif isinstance(item, dict) and "screen" in item:
            # 描述子形式 {"tid"/"id": ..., "screen": (x, y)}，目标可能不在
            # 默认世界里，直接用它自带的屏幕坐标参与排序。
            tid = int(item.get("tid", item.get("id")))
            screen = (float(item["screen"][0]), float(item["screen"][1]))
            target = s.targets.get(tid)
            if target is None and item.get("alive", True):
                keys.append((math.hypot(screen[0], screen[1]), tid))
                continue
            if target is None or not target.alive:
                continue
        elif isinstance(item, int):
            target = s.targets.get(item)
        elif _descriptor_parts(item) is not None:
            tid, screen = _descriptor_parts(item)  # type: ignore[misc]
            target = s.targets.get(tid)
            if target is not None and target.alive:
                # 目标在世界里时以它的实时屏幕坐标为准。
                screen = None
            else:
                keys.append((math.hypot(screen[0], screen[1]), tid))
                continue
        else:
            descriptor = _coerce_handle(item)
            target = (
                s.targets.get(descriptor.tid) if descriptor is not None else None
            )
        if target is None or not target.alive:
            continue
        if screen is not None:
            keys.append((math.hypot(screen[0], screen[1]), target.tid))
        else:
            keys.append((target.screen_distance(), target.tid))

    keys.sort(key=lambda pair: (pair[0], pair[1]))
    return [tid for _, tid in keys]


def reset_interp(
    state: Optional[LockState] = None,
    rate: Optional[float] = None,
) -> Dict[str, Any]:
    """重置镜头插值参数（在新锁定/切换时调用）。

    清零时间、累计角，并从当前镜头朝向重新开始最短弧插值，因此连续切换
    不会累积出整整一圈的绕行。
    """
    s = _as_state(state)
    # 以当前镜头朝向作为新一段插值的起点（最短弧），而不是沿用旧的
    # 时间 t 与累计角；这正是连续切换不会绕整圈的原因。
    yaw = float(s.interp.yaw)
    pitch = float(s.interp.pitch)
    s.interp = CameraInterp(
        epoch=s.interp.epoch + 1,
        t=0.0,
        rate=float(rate) if rate is not None else s.interp.rate,
        yaw=yaw,
        pitch=pitch,
        yaw_accum=0.0,
    )
    s.interp_resets += 1
    return {
        "epoch": s.interp.epoch,
        "t": s.interp.t,
        "yaw": s.interp.yaw,
        "pitch": s.interp.pitch,
        "yaw_accum": s.interp.yaw_accum,
        "rate": s.interp.rate,
    }


def query_seq(
    state: Optional[LockState] = None,
    tid: Optional[int] = None,
) -> Dict[str, Any]:
    """签发一次带单调序号与目标代际的异步视线查询。"""
    s = _as_state(state)
    if tid is None:
        if s.handle is not None:
            tid = s.handle.tid
        else:
            order = stable_tiebreak(s)
            tid = order[0] if order else None

    target = s.get_target(tid)
    if target is None or not target.alive:
        return {
            "seq": 0,
            "tid": tid,
            "generation": None,
            "frame": s.frame,
            "issued": False,
            "reason": "invalid_target",
        }

    query = s.issue_query(target.tid)
    result = query.as_dict()
    result["issued"] = True
    result["reason"] = "ok"
    return result


def _camera_snapshot(
    s: LockState,
    mode: str,
    reason: str,
    tid: Optional[int],
    target_pos: Optional[Vec3],
) -> Dict[str, Any]:
    snapshot = {
        "mode": mode,
        "locked": mode == "lock",
        "reason": reason,
        "tid": tid,
        "target_pos": target_pos,
        "camera_pos": s.camera_pos,
        "yaw": s.interp.yaw,
        "pitch": s.interp.pitch,
        "yaw_accum": s.interp.yaw_accum,
        "epoch": s.interp.epoch,
        "interp_t": s.interp.t,
        "freecam": mode == "freecam",
    }
    s.last_camera = snapshot
    return snapshot


def fallback_freecam(
    state: Optional[LockState] = None,
    reason: str = "lock_lost",
) -> Dict[str, Any]:
    """释放失效锁定并回退自由视角。

    句柄被清空，镜头不再读取任何目标变换；插值从当前朝向重新开始。
    """
    s = _as_state(state)
    released = s.handle
    s.handle = None
    s.freecam = True
    s.fallback_count += 1

    # 从当前镜头朝向平滑过渡到自由视角，不再读取任何目标节点。
    s.interp = CameraInterp(
        epoch=s.interp.epoch + 1,
        t=0.0,
        rate=s.interp.rate,
        yaw=s.interp.yaw,
        pitch=s.interp.pitch,
        yaw_accum=0.0,
    )

    snapshot = _camera_snapshot(s, "freecam", reason, None, None)
    snapshot["released_tid"] = released.tid if released is not None else None
    snapshot["fallback_count"] = s.fallback_count
    return snapshot


def acquire(
    state: Optional[LockState] = None,
    tid: Optional[int] = None,
    candidates: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """获取锁定：稳定排序选目标、保存代际、重置插值并发起序号查询。"""
    s, candidates = _split_state_candidates(state, candidates)

    if tid is not None:
        target = s.targets.get(tid)
        if target is None or not target.alive:
            return {
                "acquired": False,
                "reason": "target_unavailable",
                "tid": tid,
                "locked": s.handle is not None,
                "switch": False,
            }
        chosen = target
    else:
        if candidates is None:
            order = stable_tiebreak(s)
        else:
            order = stable_tiebreak(s, candidates)
        if not order:
            return {
                "acquired": False,
                "reason": "no_candidates",
                "tid": None,
                "locked": s.handle is not None,
                "switch": False,
            }
        best = order[0]
        chosen = s.targets.get(best)
        if chosen is None or not chosen.alive:
            # 候选只以描述子形式给出、世界里没有该目标：不冒险锁定。
            return {
                "acquired": False,
                "reason": "target_unavailable",
                "tid": best,
                "locked": s.handle is not None,
                "switch": False,
            }

    new_handle = LockHandle(chosen.tid, chosen.generation)
    switched = s.handle is not None and s.handle != new_handle
    if switched:
        s.switches += 1

    s.handle = new_handle
    s.freecam = False
    s.acquires += 1

    # 关键修复：每次（重新）锁定都重置插值，不再沿用上一次切换的
    # 时间/累计角，避免连续切换后镜头绕行一圈。
    interp = reset_interp(s)
    query = s.issue_query(chosen.tid)

    snapshot = _camera_snapshot(
        s, "lock", "acquired", chosen.tid, chosen.transform()
    )
    snapshot.update(
        {
            "acquired": True,
            "tid": chosen.tid,
            "generation": chosen.generation,
            "switch": switched,
            "query_seq": query.seq,
            "interp_epoch": interp["epoch"],
            "handle": {"tid": chosen.tid, "generation": chosen.generation},
        }
    )
    return snapshot


def release(
    state: Optional[LockState] = None,
    reason: str = "player_release",
) -> Dict[str, Any]:
    """主动解除锁定并回退自由视角。"""
    s = _as_state(state)
    released_tid = s.handle.tid if s.handle is not None else None
    s.releases += 1
    snapshot = fallback_freecam(s, reason)
    snapshot["released_tid"] = released_tid
    snapshot["releases"] = s.releases
    return snapshot


def update_lock(
    state: Optional[LockState] = None,
    resolved: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """逐帧更新锁定：代际校验、合并异步查询、重新做视线检查。

    ``resolved`` 是若干 ``(seq, visible)``（或含 ``seq``/``visible`` 的字典），
    按到达顺序传入；内部按序号与代际过滤，旧结果无法覆盖新结果。
    """
    s = _as_state(state)

    if isinstance(resolved, dict) or (
        isinstance(resolved, tuple) and len(resolved) == 2
        and not isinstance(resolved[0], (dict, tuple))
    ):
        # 允许只传一个已完成的结果。
        resolved = [resolved]

    s.frame += 1

    # handle 为 None 才是“没有锁定”的唯一权威状态；freecam 只是它的投影，
    # 因此即使调用方手工塞了一个失效句柄，这里也照样会校验并回退。
    if s.handle is None:
        s.freecam = True
        snapshot = _camera_snapshot(s, "freecam", "no_lock", None, None)
        snapshot.update(
            {
                "frame": s.frame,
                "locked": False,
                "visible": None,
                "los_checks": s.los_checks,
            }
        )
        return snapshot

    check = handle_generation(s, s.handle)
    if not check["valid"]:
        snapshot = fallback_freecam(s, check["reason"])
        snapshot.update(
            {
                "frame": s.frame,
                "locked": False,
                "visible": None,
                "los_checks": s.los_checks,
            }
        )
        return snapshot

    tid = s.handle.tid
    target = s.targets[tid]
    if target.transform() is None:
        snapshot = fallback_freecam(s, "target_destroyed")
        snapshot.update(
            {
                "frame": s.frame,
                "locked": False,
                "visible": None,
                "los_checks": s.los_checks,
            }
        )
        return snapshot

    for item in resolved or []:
        if isinstance(item, dict):
            seq = int(item["seq"])
            visible = bool(item.get("visible"))
        else:
            seq, visible = item
            seq = int(seq)
            visible = bool(visible)
        s.resolve_query(seq, visible)

    # 逐帧重做视线：目标移动/进入掩体当帧即可感知。
    visible = s.check_los(tid)
    async_visible = s.latest_async_visible(tid)

    if not visible or async_visible is False:
        snapshot = fallback_freecam(s, "los_lost")
        snapshot.update(
            {
                "frame": s.frame,
                "locked": False,
                "visible": visible,
                "async_visible": async_visible,
                "los_checks": s.los_checks,
            }
        )
        return snapshot

    snapshot = _camera_snapshot(s, "lock", "locked", tid, target.pos)
    snapshot.update(
        {
            "frame": s.frame,
            "locked": True,
            "visible": visible,
            "async_visible": async_visible,
            "generation": target.generation,
            "applied_query_seq": s.applied_seq.get(tid),
            "los_checks": s.los_checks,
            "stale_results_dropped": s.stale_results_dropped,
        }
    )
    return snapshot


def camera_lock(
    state: Optional[LockState] = None,
    dt: float = 1.0 / 60.0,
) -> Dict[str, Any]:
    """镜头软锁：仅从仍然存活且代际匹配的目标读取位置并插值。

    句柄失效时直接回退自由视角，不会对已释放节点求角度、也就不会得到零点。
    """
    s = _as_state(state)

    if s.handle is None:
        return fallback_freecam(s, "no_lock")

    check = handle_generation(s, s.handle)
    if not check["valid"]:
        return fallback_freecam(s, check["reason"])

    tid = s.handle.tid
    target = s.targets.get(tid)
    if target is None:
        return fallback_freecam(s, "missing_target")
    transform = target.transform()
    if transform is None:
        return fallback_freecam(s, "target_destroyed")

    desired_yaw, desired_pitch = _camera_angles(s.camera_pos, transform)

    # 最短弧插值：delta 始终在 (-180, 180]，且 epoch 已在切换时重置。
    delta_yaw = _wrap_deg(desired_yaw - s.interp.yaw)
    delta_pitch = desired_pitch - s.interp.pitch
    k = 1.0 - math.exp(-s.interp.rate * max(0.0, dt))
    step_yaw = delta_yaw * k

    s.interp.yaw = _wrap_deg(s.interp.yaw + step_yaw)
    s.interp.pitch = s.interp.pitch + delta_pitch * k
    s.interp.yaw_accum += abs(step_yaw)
    s.interp.t += max(0.0, dt)

    snapshot = _camera_snapshot(s, "lock", "tracking", tid, transform)
    snapshot.update(
        {
            "target_yaw": desired_yaw,
            "target_pitch": desired_pitch,
            "delta_yaw": delta_yaw,
            "locked": True,
        }
    )
    return snapshot


def validate_target(
    state: Optional[LockState] = None,
    tid: Optional[int] = None,
) -> Dict[str, Any]:
    """校验目标本身：存在、存活、且代际不早于当前句柄。"""
    s = _as_state(state)
    if tid is None and s.handle is not None:
        tid = s.handle.tid

    target = s.get_target(tid)
    if target is None:
        return {
            "valid": False,
            "reason": "missing_target",
            "tid": tid,
            "alive": False,
            "generation": None,
        }

    stale_generation = (
        s.handle is not None
        and s.handle.tid == tid
        and s.handle.generation != target.generation
    )
    if not target.alive:
        return {
            "valid": False,
            "reason": "target_destroyed",
            "tid": tid,
            "alive": False,
            "generation": target.generation,
        }
    if stale_generation:
        return {
            "valid": False,
            "reason": "stale_generation",
            "tid": tid,
            "alive": True,
            "generation": target.generation,
            "handle_generation": s.handle.generation,
        }
    return {
        "valid": True,
        "reason": "ok",
        "tid": tid,
        "alive": True,
        "generation": target.generation,
        "transform_available": True,
    }


def metrics(state: Optional[LockState] = None) -> Dict[str, Any]:
    """返回锁定系统的运行计数，便于审计与回归测试。"""
    s = _as_state(state)
    return {
        "frame": s.frame,
        "locked": s.handle is not None and not s.freecam,
        "locked_tid": s.handle.tid if s.handle is not None else None,
        "los_checks": s.los_checks,
        "los_failures": s.los_failures,
        "los_checked_this_frame": bool(
            s.los_log and s.los_log[-1][0] == s.frame
        ),
        "acquires": s.acquires,
        "releases": s.releases,
        "switches": s.switches,
        "interp_resets": s.interp_resets,
        "fallbacks": s.fallback_count,
        "queries_issued": s.query_issues,
        "stale_results_dropped": s.stale_results_dropped,
        "latest_applied_seq": dict(s.applied_seq),
        "interp_epoch": s.interp.epoch,
        "yaw": s.interp.yaw,
        "pitch": s.interp.pitch,
        "yaw_accum": s.interp.yaw_accum,
        "target_count": len(s.targets),
        "alive_target_count": len(s.live_candidates()),
    }


def audit(state: Optional[LockState] = None) -> Dict[str, Any]:
    """运行六个回归场景，验证题目中列出的所有缺陷均已修复。"""
    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # 1) 视线检查每帧重做：掩体出现后的下一帧立刻判定不可见并掉锁。
    s1 = _default_state()
    acquire(s1, tid=3)
    first = los_every_frame(s1)
    s1.occluders = [Occluder(center=(0.0, 0.0, -5.0), radius=2.0)]
    second = los_every_frame(s1)
    # las_every_frame 已在这帧判定不可见并释放锁定，此处应回退自由视角。
    updated = update_lock(s1)
    record(
        "los_every_frame",
        (
            first["visible"] is True
            and second["visible"] is False
            and second["released"] is True
            and second["frame"] > first["frame"]
            and s1.los_checks >= 2
            and s1.handle is None
            and updated["mode"] == "freecam"
            and updated["reason"] == "no_lock"
        ),
        (
            "visible=%s then %s, checks=%d, lock_released=%s"
            % (first["visible"], second["visible"], s1.los_checks, s1.handle is None)
        ),
    )

    # 2) 句柄代际校验：销毁目标后旧句柄失效，镜头回退自由视角且不读零点。
    s2 = _default_state()
    acquire(s2, tid=3)
    handle = s2.handle
    s2.targets[3].destroy()
    generation = handle_generation(s2, handle)
    camera = camera_lock(s2)
    record(
        "handle_generation",
        (
            not generation["valid"]
            and generation["reason"] == "target_destroyed"
            and camera["mode"] == "freecam"
            and camera["target_pos"] is None
            and s2.handle is None
        ),
        (
            "generation=%s/%s, camera_mode=%s, target_pos=%s"
            % (
                generation["reason"],
                generation["current_generation"],
                camera["mode"],
                camera["target_pos"],
            )
        ),
    )

    # 3) 同距按稳定 id 排序：不同插入顺序下结果一致。
    s3a = _default_state(seed=0)
    s3b = _default_state(seed=1)
    order_a = stable_tiebreak(s3a)
    order_b = stable_tiebreak(s3b)
    chosen_a = acquire(s3a)["tid"]
    chosen_b = acquire(s3b)["tid"]
    record(
        "stable_tiebreak",
        order_a == order_b == [3, 7] and chosen_a == chosen_b == 3,
        "order_a=%s order_b=%s chosen=%s/%s" % (
            order_a, order_b, chosen_a, chosen_b
        ),
    )

    # 4) 切换重置插值：连续切换 epoch 严格递增，且每段累计角不超过半圈。
    s4 = LockState()
    s4.targets = {
        1: Target(1, pos=(10.0, 0.0, -10.0), screen=(100.0, 100.0)),
        2: Target(2, pos=(-10.0, 0.0, -10.0), screen=(100.0, 100.0)),
    }
    sweep_ok = True
    epochs: List[int] = []
    for tid in (1, 2, 1, 2):
        before_epoch = s4.interp.epoch
        acquire(s4, tid=tid)
        epochs.append(s4.interp.epoch)
        for _ in range(90):
            camera_lock(s4, dt=1.0 / 30.0)
        if s4.interp.epoch != before_epoch + 1:
            sweep_ok = False
        if s4.interp.yaw_accum > MAX_SWITCH_SWEEP_DEG + 0.01:
            sweep_ok = False
    record(
        "reset_interp",
        sweep_ok and s4.switches == 3,
        "epochs=%s switches=%d last_yaw_accum=%.3f"
        % (epochs, s4.switches, s4.interp.yaw_accum),
    )

    # 5) 查询序号：旧查询晚到不覆盖新结果；目标代际变化后的结果也丢弃。
    s5 = _default_state()
    acquire(s5, tid=3)
    old_query = query_seq(s5, tid=3)
    new_query = query_seq(s5, tid=3)
    applied_new = s5.resolve_query(new_query["seq"], True)
    applied_old = s5.resolve_query(old_query["seq"], False)
    stale_ok = (
        applied_new["applied"] is True
        and applied_old["applied"] is False
        and applied_old["reason"] == "stale_sequence"
        and s5.latest_async_visible(3) is True
    )

    s5b = _default_state()
    acquire(s5b, tid=3)
    late_query = query_seq(s5b, tid=3)
    s5b.targets[3].destroy()
    stale_target = s5b.resolve_query(late_query["seq"], True)
    stale_ok = stale_ok and (
        stale_target["applied"] is False
        and stale_target["reason"] == "stale_target"
    )
    record(
        "query_seq",
        stale_ok,
        "old=%s new_applied=%s latest=%s stale_target=%s"
        % (
            applied_old["reason"],
            applied_new["applied"],
            s5.latest_async_visible(3),
            stale_target["reason"],
        ),
    )

    # 6) 锁定丢失回退自由视角（上面的场景已覆盖视线与销毁，这里覆盖主动释放）。
    s6 = _default_state()
    acquire(s6)
    released = release(s6, reason="lock_lost")
    record(
        "fallback_freecam",
        (
            released["mode"] == "freecam"
            and released["locked"] is False
            and released["target_pos"] is None
            and s6.handle is None
            and s6.freecam is True
        ),
        "mode=%s reason=%s released_tid=%s"
        % (released["mode"], released["reason"], released["released_tid"]),
    )

    ok = all(item["ok"] for item in checks)
    report = {"ok": ok, "checks": checks}
    if isinstance(state, LockState):
        report["metrics"] = metrics(state)
    return report


def fallback(
    state: Optional[LockState] = None,
    reason: str = "lock_lost",
) -> Dict[str, Any]:
    """通用回退入口：释放锁定并切换到自由视角。"""
    s = _as_state(state)
    snapshot = fallback_freecam(s, reason)
    snapshot["kind"] = "fallback"
    return snapshot


def determinism(
    state: Optional[LockState] = None,
    runs: Optional[int] = None,
) -> Dict[str, Any]:
    """重复构建同一场景并锁定，验证选择结果与排序完全确定。"""
    custom_candidates: Optional[Iterable[Any]] = None
    if isinstance(state, int):
        run_count = state
    elif runs is not None:
        run_count = int(runs)
    elif isinstance(state, (list, tuple, set, dict)):
        # 也允许传入自定义候选集合，验证其排序是否稳定。
        custom_candidates = state
        run_count = 8
    else:
        run_count = 8
    run_count = max(1, run_count)

    choices: List[Optional[int]] = []
    orders: List[List[int]] = []
    for index in range(run_count):
        s = _default_state(seed=index)
        if custom_candidates is not None:
            # 用不同的插入顺序洗一遍候选集合，排序仍须一致。
            items = list(custom_candidates)
            if index % 2:
                items = items[::-1]
            order = stable_tiebreak(s, items)
        else:
            order = stable_tiebreak(s)
        orders.append(order)
        chosen_tid: Optional[int] = None
        if order:
            if custom_candidates is None:
                chosen_tid = acquire(s)["tid"]
            else:
                # 目标未必在默认世界里，描述子自己就带排序键。
                chosen_tid = order[0]
        choices.append(chosen_tid)

    stable_choices = len({choice for choice in choices}) == 1
    stable_orders = len({tuple(order) for order in orders}) == 1
    return {
        "deterministic": bool(stable_choices and stable_orders),
        "runs": run_count,
        "choices": choices,
        "order": orders[0] if orders else [],
        "chosen": choices[0] if choices else None,
    }
