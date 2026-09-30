// SPDX-License-Identifier: BSD-3-Clause
// Read-only PhysX 5.9 ABI bridge used by the kit-less Contact C0 worker.

#include "PxArticulationLink.h"
#include "PxScene.h"
#include "foundation/PxPhysicsVersion.h"

#include <cmath>
#include <cstdint>

extern "C" {

std::uint32_t waveqa_physx_version() {
    return static_cast<std::uint32_t>(PX_PHYSICS_VERSION);
}

std::uint32_t waveqa_scene_ccd_mask() {
    return static_cast<std::uint32_t>(physx::PxSceneFlag::eENABLE_CCD);
}

std::uint32_t waveqa_rigid_body_ccd_mask() {
    return static_cast<std::uint32_t>(physx::PxRigidBodyFlag::eENABLE_CCD);
}

std::uint32_t waveqa_scene_flags(const void* scene_pointer) {
    if (scene_pointer == nullptr) {
        return UINT32_MAX;
    }
    const auto* scene = static_cast<const physx::PxScene*>(scene_pointer);
    return static_cast<std::uint32_t>(scene->getFlags());
}

std::uint32_t waveqa_rigid_body_flags(const void* body_pointer) {
    if (body_pointer == nullptr) {
        return UINT32_MAX;
    }
    const auto* body = static_cast<const physx::PxArticulationLink*>(body_pointer);
    return static_cast<std::uint32_t>(body->getRigidBodyFlags());
}

std::uint32_t waveqa_mass_properties_value_count() {
    return 11U;
}

std::uint32_t waveqa_rigid_body_mass_properties(
    const void* body_pointer,
    double* output_values,
    std::uint32_t output_value_count) {
    if (body_pointer == nullptr) {
        return 1U;
    }
    if (output_values == nullptr) {
        return 2U;
    }
    if (output_value_count != waveqa_mass_properties_value_count()) {
        return 3U;
    }

    // OVPhysX PhysXType.LINK returns a PxArticulationLink pointer.  Preserve
    // that exact dynamic type rather than assuming that the address can be
    // reinterpreted directly as one of its base classes.
    const auto* body = static_cast<const physx::PxArticulationLink*>(body_pointer);
    const physx::PxReal mass = body->getMass();
    const physx::PxVec3 inertia = body->getMassSpaceInertiaTensor();
    const physx::PxTransform center_of_mass = body->getCMassLocalPose();
    const double values[11] = {
        static_cast<double>(mass),
        static_cast<double>(inertia.x),
        static_cast<double>(inertia.y),
        static_cast<double>(inertia.z),
        static_cast<double>(center_of_mass.p.x),
        static_cast<double>(center_of_mass.p.y),
        static_cast<double>(center_of_mass.p.z),
        static_cast<double>(center_of_mass.q.x),
        static_cast<double>(center_of_mass.q.y),
        static_cast<double>(center_of_mass.q.z),
        static_cast<double>(center_of_mass.q.w),
    };
    for (const double value : values) {
        if (!std::isfinite(value)) {
            return 4U;
        }
    }
    if (values[0] <= 0.0 || values[1] <= 0.0 || values[2] <= 0.0 ||
        values[3] <= 0.0) {
        return 5U;
    }
    const double quaternion_norm_squared =
        values[7] * values[7] + values[8] * values[8] +
        values[9] * values[9] + values[10] * values[10];
    if (quaternion_norm_squared <= 0.0 ||
        std::abs(std::sqrt(quaternion_norm_squared) - 1.0) > 1.0e-5) {
        return 6U;
    }
    for (std::uint32_t index = 0; index < output_value_count; ++index) {
        output_values[index] = values[index];
    }
    return 0U;
}

}  // extern "C"
