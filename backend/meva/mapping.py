"""Every mapping is explicit; unsupported activities never count as successes."""

DIRECT = {
    "person_picks_up_object": (
        "picked_up",
        "Stationary-to-coupled-motion hypothesis; semantic pickup target.",
    ),
    "person_puts_down_object": (
        "placed_object",
        "Coupled motion to stationary object and sustained person separation.",
    ),
    "vehicle_starts": (
        "started_moving",
        "Stationary-to-moving transition for a vehicle.",
    ),
    "vehicle_stops": (
        "stopped",
        "Moving-to-stationary vehicle transition; duration mismatch remains measurable.",
    ),
}
APPROXIMATE = {
    "person_enters_scene_through_structure": (
        "appeared",
        "Track appearance is a weak proxy, not proof of passing through a structure; reported separately.",
    ),
    "person_exits_scene_through_structure": (
        "disappeared",
        "Track disappearance is a weak proxy; occlusion/fragmentation can create false positives; reported separately.",
    ),
}
ACTIVITIES = """hand_interacts_with_person person_abandons_package person_carries_heavy_object
person_closes_facility_door person_closes_trunk person_closes_vehicle_door person_embraces_person
person_enters_scene_through_structure person_enters_vehicle person_exits_scene_through_structure
person_exits_vehicle person_interacts_with_laptop person_loads_vehicle person_opens_facility_door
person_opens_trunk person_opens_vehicle_door person_picks_up_object person_purchases
person_puts_down_object person_reads_document person_rides_bicycle person_sits_down person_stands_up
person_steals_object person_talks_on_phone person_talks_to_person person_texts_on_phone
person_transfers_object person_unloads_vehicle vehicle_drops_off_person vehicle_makes_u_turn
vehicle_picks_up_person vehicle_reverses vehicle_starts vehicle_stops vehicle_turns_left vehicle_turns_right""".split()


def mapping(activity):
    if activity in DIRECT:
        event, reason = DIRECT[activity]
        return {"event_type": event, "status": "DIRECT", "reason": reason}
    if activity in APPROXIMATE:
        event, reason = APPROXIMATE[activity]
        return {"event_type": event, "status": "APPROXIMATE", "reason": reason}
    return {
        "event_type": None,
        "status": "UNSUPPORTED",
        "reason": "No compatible production action detector. Proximity/class detection does not establish this activity.",
    }
