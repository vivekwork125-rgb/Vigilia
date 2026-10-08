# MEVA activity mappings

These mappings are evaluation-only. DIRECT is a compatible semantic target, not a claim that VIGILIA reliably detects it. APPROXIMATE scores are separate. Unknown activities default to UNSUPPORTED.

| MEVA activity | VIGILIA target | Status | Justification |
|---|---|---|---|
| hand_interacts_with_person | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_abandons_package | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_carries_heavy_object | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_closes_facility_door | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_closes_trunk | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_closes_vehicle_door | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_embraces_person | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_enters_scene_through_structure | appeared | APPROXIMATE | Track appearance is a weak proxy, not proof of passing through a structure; reported separately. |
| person_enters_vehicle | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_exits_scene_through_structure | disappeared | APPROXIMATE | Track disappearance is a weak proxy; occlusion/fragmentation can create false positives; reported separately. |
| person_exits_vehicle | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_interacts_with_laptop | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_loads_vehicle | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_opens_facility_door | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_opens_trunk | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_opens_vehicle_door | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_picks_up_object | picked_up | DIRECT | Stationary-to-coupled-motion hypothesis; semantic pickup target. |
| person_purchases | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_puts_down_object | placed_object | DIRECT | Coupled motion to stationary object and sustained person separation. |
| person_reads_document | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_rides_bicycle | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_sits_down | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_stands_up | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_steals_object | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_talks_on_phone | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_talks_to_person | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_texts_on_phone | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_transfers_object | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| person_unloads_vehicle | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| vehicle_drops_off_person | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| vehicle_makes_u_turn | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| vehicle_picks_up_person | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| vehicle_reverses | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| vehicle_starts | started_moving | DIRECT | Stationary-to-moving transition for a vehicle. |
| vehicle_stops | stopped | DIRECT | Moving-to-stationary vehicle transition; duration mismatch remains measurable. |
| vehicle_turns_left | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
| vehicle_turns_right | — | UNSUPPORTED | No compatible production action detector. Proximity/class detection does not establish this activity. |
