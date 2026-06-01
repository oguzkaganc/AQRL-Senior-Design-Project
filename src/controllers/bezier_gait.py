"""
Bezier curve gait generator for the AQRL trot controller.

Internal convention: X=forward, Y=lateral, Z=up

Generates body-to-foot trajectory offsets using:
    - Bezier curves for swing phase (11th order, 12 control points)
    - Sinusoidal for stance phase

Phase lags in gait-generator order [FL, FR, BL, BR]:
    Trot: FL=0.0, FR=0.5, BL=0.5, BR=0.0
    (FL-BR diagonal in sync, FR-BL diagonal in sync)

The Bezier gait structure follows common open-source quadruped gait examples
and was adapted to the AQRL simulation and deployment pipeline.
"""

import numpy as np
import copy
import math

from .lie_algebra import TransToRp

STANCE = 0
SWING = 1


class BezierGait:
    def __init__(self, dSref=[0.0, 0.0, 0.5, 0.5], dt=0.01, Tswing=0.2):
        """
        :param dSref: Phase lag per leg [FL, FR, BL, BR]. FL is reference (0.0).
        :param dt: Timestep (seconds)
        :param Tswing: Swing period (seconds)
        """
        self.dSref = dSref
        self.Prev_fxyz = [0.0, 0.0, 0.0, 0.0]
        self.NumControlPoints = 11  # n+1 = 12 control points
        self.dt = dt

        self.time = 0.0
        self.TD_time = 0.0
        self.time_since_last_TD = 0.0
        self.StanceSwing = SWING
        self.SwRef = 0.0
        self.Stref = 0.0
        self.TD = False
        self.Tswing = Tswing
        self.ref_idx = 0
        self.Phases = list(self.dSref)

    def reset(self):
        """Reset gait generator state."""
        self.Prev_fxyz = [0.0, 0.0, 0.0, 0.0]
        self.time = 0.0
        self.TD_time = 0.0
        self.time_since_last_TD = 0.0
        self.StanceSwing = SWING
        self.SwRef = 0.0
        self.Stref = 0.0
        self.TD = False

    def GetPhase(self, index, Tstance, Tswing):
        """Get phase and stance/swing state for a single leg."""
        StanceSwing = STANCE
        Tstride = Tstance + Tswing
        ti = self.Get_ti(index, Tstride)

        if ti < -Tswing:
            ti += Tstride

        # STANCE
        if ti >= 0.0 and ti <= Tstance:
            StanceSwing = STANCE
            Stnphase = ti / float(Tstance) if Tstance != 0.0 else 0.0
            if index == self.ref_idx:
                self.StanceSwing = StanceSwing
            return Stnphase, StanceSwing

        # SWING
        Sw_phase = 0.0
        if ti >= -Tswing and ti < 0.0:
            StanceSwing = SWING
            Sw_phase = (ti + Tswing) / Tswing
        elif ti > Tstance and ti <= Tstride:
            StanceSwing = SWING
            Sw_phase = (ti - Tstance) / Tswing

        if Sw_phase >= 1.0:
            Sw_phase = 1.0

        if index == self.ref_idx:
            self.StanceSwing = StanceSwing
            self.SwRef = Sw_phase
            if self.SwRef >= 0.999:
                self.TD = True

        return Sw_phase, StanceSwing

    def Get_ti(self, index, Tstride):
        """Get time index for a leg, accounting for phase lag."""
        if index == self.ref_idx:
            self.dSref[index] = 0.0
        return self.time_since_last_TD - self.dSref[index] * Tstride

    def Increment(self, dt, Tstride):
        """Advance internal clock."""
        self.CheckTouchDown()
        self.time_since_last_TD = self.time - self.TD_time
        self.time_since_last_TD = np.clip(self.time_since_last_TD, 0.0, Tstride)
        self.time += dt

        # If Tstride = Tswing (Tstance = 0), reset all
        if Tstride < self.Tswing + dt:
            self.time = 0.0
            self.time_since_last_TD = 0.0
            self.TD_time = 0.0
            self.SwRef = 0.0

    def CheckTouchDown(self):
        """Check and handle reference leg touchdown."""
        if self.SwRef >= 0.9 and self.TD:
            self.TD_time = self.time
            self.TD = False
            self.SwRef = 0.0

    def BernSteinPoly(self, t, k, point):
        """Evaluate Bernstein polynomial at phase t for control point k."""
        return point * self.Binomial(k) * np.power(t, k) * np.power(
            1 - t, self.NumControlPoints - k)

    def Binomial(self, k):
        """Binomial coefficient C(n, k) where n = NumControlPoints."""
        return math.factorial(self.NumControlPoints) / (
            math.factorial(k) *
            math.factorial(self.NumControlPoints - k))

    def BezierSwing(self, phase, L, LateralFraction, clearance_height=0.04):
        """
        Bezier curve swing trajectory.

        :param phase: swing phase [0, 1]
        :param L: half step length
        :param LateralFraction: angle for lateral/forward decomposition
        :param clearance_height: max foot clearance
        :return: (stepX, stepY, stepZ) offset relative to default foot pos
        """
        X_POLAR = np.cos(LateralFraction)
        Y_POLAR = np.sin(LateralFraction)

        # 12 Bezier control points for forward component
        STEP = np.array([
            -L,  -L * 1.4,  -L * 1.5,  -L * 1.5,  -L * 1.5,
            0.0,  0.0,  0.0,
            L * 1.5,  L * 1.5,  L * 1.4,  L
        ])
        X = STEP * X_POLAR
        Y = STEP * Y_POLAR

        # 12 control points for vertical component
        Z = np.array([
            0.0, 0.0,
            clearance_height * 0.9, clearance_height * 0.9, clearance_height * 0.9,
            clearance_height * 0.9, clearance_height * 0.9,
            clearance_height * 1.1, clearance_height * 1.1, clearance_height * 1.1,
            0.0, 0.0,
        ])

        stepX = sum(self.BernSteinPoly(phase, i, X[i]) for i in range(len(X)))
        stepY = sum(self.BernSteinPoly(phase, i, Y[i]) for i in range(len(Y)))
        stepZ = sum(self.BernSteinPoly(phase, i, Z[i]) for i in range(len(Z)))

        return stepX, stepY, stepZ

    def SineStance(self, phase, L, LateralFraction, penetration_depth=0.0):
        """
        Sinusoidal stance trajectory.

        :param phase: stance phase [0, 1]
        :param L: half step length
        :param LateralFraction: angle for lateral/forward decomposition
        :param penetration_depth: foot ground penetration
        :return: (stepX, stepY, stepZ) offset
        """
        X_POLAR = np.cos(LateralFraction)
        Y_POLAR = np.sin(LateralFraction)
        step = L * (1.0 - 2.0 * phase)
        stepX = step * X_POLAR
        stepY = step * Y_POLAR
        if L != 0.0:
            stepZ = -penetration_depth * np.cos(
                (np.pi * (stepX + stepY)) / (2.0 * L))
        else:
            stepZ = 0.0
        return stepX, stepY, stepZ

    def YawCircle(self, T_bf, index):
        """Compute yaw rotation angle for foot trajectory plane."""
        DefaultBodyToFoot_Magnitude = np.sqrt(T_bf[0]**2 + T_bf[1]**2)
        DefaultBodyToFoot_Direction = np.arctan2(T_bf[1], T_bf[0])

        g_xyz = self.Prev_fxyz[index] - np.array([T_bf[0], T_bf[1], T_bf[2]])
        g_mag = np.sqrt(g_xyz[0]**2 + g_xyz[1]**2)
        th_mod = np.arctan2(g_mag, DefaultBodyToFoot_Magnitude)

        # FR and BL (index 1, 2)
        if index == 1 or index == 2:
            phi_arc = np.pi / 2.0 + DefaultBodyToFoot_Direction + th_mod
        # FL and BR (index 0, 3)
        else:
            phi_arc = np.pi / 2.0 - DefaultBodyToFoot_Direction + th_mod

        return phi_arc

    def SwingStep(self, phase, L, LateralFraction, YawRate,
                  clearance_height, T_bf, key, index):
        """Combined forward + rotational swing step.

        Z clearance is added once (linear component only) — the yaw term
        contributes XY rotation on the ground, not extra lift.
        """
        phi_arc = self.YawCircle(T_bf, index)
        X_lin, Y_lin, Z_lin = self.BezierSwing(phase, L, LateralFraction, clearance_height)
        X_rot, Y_rot, _ = self.BezierSwing(phase, YawRate, phi_arc, clearance_height)
        coord = np.array([X_lin + X_rot, Y_lin + Y_rot, Z_lin])
        self.Prev_fxyz[index] = coord
        return coord

    def StanceStep(self, phase, L, LateralFraction, YawRate,
                   penetration_depth, T_bf, key, index):
        """Combined forward + rotational stance step.

        Z penetration depth added once (linear component only).
        """
        phi_arc = self.YawCircle(T_bf, index)
        X_lin, Y_lin, Z_lin = self.SineStance(phase, L, LateralFraction, penetration_depth)
        X_rot, Y_rot, _ = self.SineStance(phase, YawRate, phi_arc, penetration_depth)
        coord = np.array([X_lin + X_rot, Y_lin + Y_rot, Z_lin])
        self.Prev_fxyz[index] = coord
        return coord

    def GetFootStep(self, L, LateralFraction, YawRate, clearance_height,
                    penetration_depth, Tstance, T_bf, index, key):
        """Get foot step coordinates based on current phase."""
        phase, StanceSwing = self.GetPhase(index, Tstance, self.Tswing)
        self.Phases[index] = phase + 1.0 if StanceSwing == SWING else phase

        if StanceSwing == STANCE:
            return self.StanceStep(phase, L, LateralFraction, YawRate,
                                   penetration_depth, T_bf, key, index)
        else:
            return self.SwingStep(phase, L, LateralFraction, YawRate,
                                  clearance_height, T_bf, key, index)

    def GenerateTrajectory(self, L, LateralFraction, YawRate, vel,
                           T_bf_,
                           clearance_height=0.06, penetration_depth=0.01,
                           contacts=None, dt=None):
        """
        Generate foot trajectories for all 4 legs.

        :param L: half step length
        :param LateralFraction: lateral movement angle
        :param YawRate: desired yaw rate
        :param vel: desired step velocity
        :param T_bf_: default body-to-foot transforms (OrderedDict)
        :param clearance_height: swing foot clearance
        :param penetration_depth: stance foot penetration
        :param contacts: [4] contact booleans
        :param dt: timestep override
        :return: Updated T_bf OrderedDict with trajectory offsets applied
        """
        # Compute Tstance from desired speed
        if vel != 0.0:
            Tstance = 2.0 * abs(L) / abs(vel)
        else:
            Tstance = 0.0
            L = 0.0
            self.TD = False
            self.time = 0.0
            self.time_since_last_TD = 0.0

        if dt is None:
            dt = self.dt

        YawRate *= dt

        # Catch infeasible timesteps
        if Tstance < dt:
            Tstance = 0.0
            L = 0.0
            self.TD = False
            self.time = 0.0
            self.time_since_last_TD = 0.0
            YawRate = 0.0
        elif Tstance > 1.3 * self.Tswing:
            Tstance = 1.3 * self.Tswing

        if contacts is None:
            contacts = [0, 0, 0, 0]

        # Check contacts
        if contacts[0] == 1 and Tstance > dt:
            self.TD = True

        self.Increment(dt, Tstance + self.Tswing)

        T_bf = copy.deepcopy(T_bf_)
        for i, (key, Tbf_in) in enumerate(T_bf_.items()):
            # Set phase lags (trot pattern)
            if key == "FL":
                self.ref_idx = i
                self.dSref[i] = 0.0
            elif key == "FR":
                self.dSref[i] = 0.5
            elif key == "BL":
                self.dSref[i] = 0.5
            elif key == "BR":
                self.dSref[i] = 0.0

            _, p_bf = TransToRp(Tbf_in)
            if Tstance > 0.0:
                step_coord = self.GetFootStep(
                    L, LateralFraction, YawRate, clearance_height,
                    penetration_depth, Tstance, p_bf, i, key)
            else:
                step_coord = np.array([0.0, 0.0, 0.0])

            T_bf[key][0, 3] = Tbf_in[0, 3] + step_coord[0]
            T_bf[key][1, 3] = Tbf_in[1, 3] + step_coord[1]
            T_bf[key][2, 3] = Tbf_in[2, 3] + step_coord[2]

        return T_bf
