"""Enumerations for travel diary survey processing."""

from enum import IntEnum, StrEnum


class LocationType(StrEnum):
    """Location type classifications for trip origins and destinations."""

    HOME = "home"
    WORK = "work"
    SCHOOL = "school"
    OTHER = "other"


class TourCategory(StrEnum):
    """Tour category classifications."""

    HOME_BASED = "home_based"
    WORK_BASED = "work_based"


class HalfTour(StrEnum):
    """Half-tour direction classifications."""

    OUTBOUND = "outbound"
    INBOUND = "inbound"


class PersonCategory(StrEnum):
    """Person category for purpose priority determination."""

    WORKER = "worker"
    STUDENT = "student"
    OTHER = "other"


class PersonType(IntEnum):
    """Standard person type codes."""

    FULL_TIME_WORKER = 1
    PART_TIME_WORKER = 2
    RETIRED = 3
    NON_WORKER = 4
    UNIVERSITY_STUDENT = 5
    HIGH_SCHOOL_STUDENT = 6
    CHILD_5_15 = 7
    CHILD_UNDER_5 = 8


class TripPurpose(IntEnum):
    """Standard trip purpose codes."""

    WORK = 1
    SCHOOL = 2
    ESCORT = 3
    SHOPPING = 4
    MEAL = 5
    SOCIAL = 6
    RECREATION = 7
    MEDICAL = 8
    OTHER = 9


class ModeType(IntEnum):
    """Standard mode type codes."""

    WALK = 1
    BIKE = 2
    AUTO = 3
    TRANSIT = 6
    DRIVE_TRANSIT = 7
