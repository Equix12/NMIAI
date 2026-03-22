"""Deterministic Tripletex action wrappers.

The LLM calls these high-level actions with extracted values.
Each action knows exactly what API calls to make and what fields to use.
No guessing, no wrong field names.
"""

import logging
from tripletex_client import TripletexClient

# Session-level cache to avoid redundant GET calls
# Each task gets its own cache via threading.local to avoid concurrent corruption
import threading
_thread_local = threading.local()


def _get_cache() -> dict:
    if not hasattr(_thread_local, "cache"):
        _thread_local.cache = {}
    return _thread_local.cache


def _reset_cache():
    """Reset cache for new submission."""
    _thread_local.cache = {}


# Backward compat: _cache property for existing code
class _CacheProxy(dict):
    """Proxy that delegates to thread-local cache."""
    def __getitem__(self, key): return _get_cache()[key]
    def __setitem__(self, key, val): _get_cache()[key] = val
    def __contains__(self, key): return key in _get_cache()
    def __delitem__(self, key): del _get_cache()[key]
    def get(self, key, default=None): return _get_cache().get(key, default)
    def setdefault(self, key, default=None): return _get_cache().setdefault(key, default)
    def pop(self, key, *args): return _get_cache().pop(key, *args)


_cache = _CacheProxy()


def _add_extra_fields(body: dict, kwargs: dict, schema_name: str | None = None) -> None:
    """Add extra fields from kwargs to the API body, using the OpenAPI schema for type coercion.

    Philosophy: pass through everything the LLM sends. The API will ignore unknown fields.
    The schema is used only for *type transformation* (e.g. wrapping IDs as {id: N}),
    never for filtering. Dropping valid fields silently is worse than sending extras.
    """
    skip = {'tx', 'costs', 'lines', 'postings', 'searchParams',
            'customerId', 'employeeId', 'projectId', 'supplierId',
            'departmentId', 'projectManagerId', 'contactId',
            'startDate', 'priceExcludingVat', 'costPrice',
            'categoryKeyword', 'activityName', 'travelExpenseId',
            'accountNumber', 'amountInclVat', 'invoiceNumber',
            'fileContent', 'filename', 'fileFormat', 'bankId',
            'accountId', 'reconciliationId', 'entityType', 'entityId',
            'paymentDate', 'paymentAmount', 'invoiceId',
            'destination', 'departureFrom', 'isForeignTravel',
            'perDiems', 'mileageAllowances', 'accommodationAllowances',
            'salaryLines', 'baseSalary', 'vatTypeId',
            'productUnit', 'supplierId', 'kid', 'invoiceRemark', 'invoiceComment',
            'projectCategoryId', 'hourlyRate', 'chargeable', 'departmentManagerId',
            'paymentTypeId', 'currency', 'deliveryAddress', 'departureDate', 'returnDate'}

    extra = {k: v for k, v in kwargs.items()
             if k not in skip and k not in body and v is not None and v != ""}

    if not extra:
        return

    # Load schema if available — used for type coercion only
    schema_fields = {}
    if schema_name:
        try:
            from api_reference import get_schema
            schema = get_schema(schema_name)
            schema_fields = schema.get("fields", {})
        except Exception:
            pass

    for field_name, value in extra.items():
        field_info = schema_fields.get(field_name, {})
        field_type = field_info.get("type", "")

        if field_type and "object" in field_type and "id" in field_type:
            # Schema says this is an object reference — wrap as {id: N}
            if isinstance(value, (int, float)):
                body[field_name] = {"id": int(value)}
            elif isinstance(value, dict) and "id" in value:
                body[field_name] = value
            elif isinstance(value, str) and "country" in field_name.lower():
                body[field_name] = {"id": _get_country_id(value)}
            else:
                # Can't transform — pass through as-is, API may still accept it
                body[field_name] = value

        elif field_type == "string":
            body[field_name] = str(value)

        elif field_type == "number":
            try:
                body[field_name] = float(value)
            except (ValueError, TypeError):
                body[field_name] = value  # pass through anyway

        elif field_type == "integer":
            try:
                body[field_name] = int(value)
            except (ValueError, TypeError):
                body[field_name] = value

        elif field_type == "boolean":
            if isinstance(value, bool):
                body[field_name] = value
            elif isinstance(value, str):
                body[field_name] = value.lower() in ("true", "1", "yes", "ja")
            else:
                body[field_name] = bool(value)

        else:
            # No schema type info, or complex type — pass through as-is
            body[field_name] = value

        if field_name in body:
            logger.info("Extra field '%s' = %s", field_name, str(body[field_name])[:80])


def _get_id(result: dict) -> int | None:
    """Safely extract ID from API result. Returns None if error or missing."""
    if result.get("_is_error"):
        return None
    value = result.get("value")
    if isinstance(value, dict):
        return value.get("id")
    return None


def _lookup_dimension_value(tx, name: str) -> int | None:
    """Look up a custom dimension value ID by its displayName/number.

    Always refreshes from API, then falls back to session cache and partial matching.
    """
    target = name.strip().lower()

    # 1. Fresh API lookup
    resp = tx.get("ledger/accountingDimensionValue", {"count": 1000})
    vals = resp.get("values", [])
    mapping = {}
    for v in vals:
        for field in ("displayName", "number"):
            key = (v.get(field) or "").strip().lower()
            if key:
                mapping[key] = v["id"]
    _cache.setdefault("dimension_values_session", {}).update(mapping)
    _cache["dimension_values"] = mapping

    # Exact match
    if target in mapping:
        return mapping[target]

    # Partial/substring match
    for key, val_id in mapping.items():
        if target in key or key in target:
            return val_id

    # 2. Session cache (picks up values created earlier this session)
    session_map = _cache.get("dimension_values_session", {})
    if target in session_map:
        return session_map[target]
    for key, val_id in session_map.items():
        if target in key or key in target:
            return val_id

    return None


logger = logging.getLogger(__name__)

# Country name → ID mapping
COUNTRY_IDS = {
    # Nordics
    "norge": 161, "norway": 161, "noruega": 161, "norwegen": 161, "norvège": 161,
    "sverige": 191, "sweden": 191, "suecia": 191, "schweden": 191, "suède": 191,
    "danmark": 57, "denmark": 57, "dinamarca": 57, "dänemark": 57, "danemark": 57,
    "finland": 68, "finnland": 68, "finlandia": 68, "finlande": 68,
    "island": 105, "iceland": 105, "islandia": 105,
    # Western Europe
    "tyskland": 55, "germany": 55, "alemania": 55, "deutschland": 55, "allemagne": 55,
    "frankrike": 73, "france": 73, "francia": 73, "frankreich": 73,
    "spania": 66, "spain": 66, "españa": 66, "spanien": 66, "espagne": 66,
    "storbritannia": 75, "uk": 75, "united kingdom": 75, "großbritannien": 75, "royaume-uni": 75,
    "portugal": 178,
    "italia": 106, "italy": 106, "italien": 106, "italie": 106,
    "nederland": 155, "netherlands": 155, "países bajos": 155, "niederlande": 155, "pays-bas": 155,
    "belgia": 21, "belgium": 21, "belgien": 21, "belgique": 21, "bélgica": 21,
    "sveits": 42, "switzerland": 42, "schweiz": 42, "suisse": 42, "suiza": 42,
    "østerrike": 13, "austria": 13, "österreich": 13, "autriche": 13,
    "luxemburg": 129, "luxembourg": 129,
    "irland": 99, "ireland": 99, "irlanda": 99, "irlande": 99,
    # Eastern Europe
    "polen": 176, "poland": 176, "polonia": 176, "pologne": 176,
    "tsjekkia": 54, "czech republic": 54, "czechia": 54, "república checa": 54,
    "ungarn": 97, "hungary": 97, "hungría": 97, "hongrie": 97,
    "romania": 183, "rumänien": 183, "roumanie": 183,
    "kroatia": 95, "croatia": 95, "croacia": 95, "croatie": 95,
    "estland": 62, "estonia": 62, "estonie": 62,
    "latvia": 130, "lettland": 130, "lettonie": 130,
    "litauen": 128, "lithuania": 128, "lituanie": 128,
    "hellas": 86, "greece": 86, "grecia": 86, "grèce": 86, "griechenland": 86,
    "tyrkia": 217, "turkey": 217, "turquía": 217, "türkei": 217, "turquie": 217,
    # Americas
    "usa": 225, "united states": 225, "estados unidos": 225, "états-unis": 225,
    "canada": 37, "canadá": 37, "kanada": 37,
    "brasil": 30, "brazil": 30, "brasilien": 30, "brésil": 30,
    "mexico": 151, "méxico": 151, "mexiko": 151, "mexique": 151,
    "argentina": 11, "argentinien": 11, "argentine": 11,
    "chile": 45,
    "colombia": 48, "kolumbien": 48, "colombie": 48,
    "peru": 168, "perú": 168, "pérou": 168,
    # Asia/Oceania
    "japan": 109, "japón": 109, "japon": 109,
    "kina": 47, "china": 47, "chine": 47,
    "india": 101, "indien": 101, "inde": 101,
    "australia": 14, "australien": 14, "australie": 14,
    "new zealand": 165, "nueva zelanda": 165, "neuseeland": 165, "nouvelle-zélande": 165,
}


def _infer_country_from_name(entity_name: str | None) -> int | None:
    """Infer country ID from organization name suffix (e.g. AS=Norway, GmbH=Germany, Ltd=UK)."""
    if not entity_name:
        return None
    name = entity_name.strip()
    # Check suffix patterns — match the last word or known patterns
    suffixes = {
        # Norwegian
        "AS": 161, "ASA": 161, "ANS": 161, "DA": 161, "NUF": 161,
        "KS": 161, "ENK": 161, "SA": 161,
        # German
        "GmbH": 55, "AG": 55, "KG": 55, "OHG": 55, "UG": 55, "e.V.": 55,
        # Swedish
        "AB": 191,
        # Danish
        "A/S": 57, "ApS": 57, "I/S": 57,
        # Finnish
        "Oy": 68, "Oyj": 68,
        # UK/English
        "Ltd": 75, "PLC": 75, "LLP": 75,
        # Spanish
        "SL": 66, "SA": 66,
        # French
        "SARL": 73, "SAS": 73, "EURL": 73,
        # Portuguese
        "Lda": 178, "LTDA": 178,
        # Dutch
        "BV": 155, "NV": 155,
        # Italian
        "SRL": 106, "SpA": 106,
    }
    # Check last word
    parts = name.split()
    if parts:
        last = parts[-1].rstrip(".")
        for suffix, country_id in suffixes.items():
            if last == suffix or last == suffix.upper() or name.endswith(f" {suffix}"):
                return country_id
    return None


def _get_country_id(country_name: str | None, city: str | None = None, postalCode: str | None = None, entity_name: str | None = None, email: str | None = None, organizationNumber: str | None = None) -> int:
    """Convert country name to Tripletex country ID. Infers from context if no country given."""
    if country_name:
        return COUNTRY_IDS.get(country_name.lower().strip(), 161)

    # Norwegian org numbers are 9 digits with mod11 check — strongest signal
    if organizationNumber:
        org = organizationNumber.strip().replace(" ", "")
        if len(org) == 9 and org.isdigit():
            weights = [3, 2, 7, 6, 5, 4, 3, 2]
            digits = [int(d) for d in org]
            total = sum(d * w for d, w in zip(digits[:8], weights))
            remainder = total % 11
            check = 0 if remainder == 0 else 11 - remainder
            if check < 10 and check == digits[8]:
                return 161  # Valid Norwegian org number

    # Infer from postal code format
    if postalCode:
        pc = postalCode.strip()
        if len(pc) == 4 and pc.isdigit():
            return 161  # Norway: 4-digit numeric
        if len(pc) == 5 and pc[:3].isdigit() and pc[3] == " " and pc[4:].isdigit():
            return 191  # Sweden: NNN NN
        if len(pc) == 5 and pc.isdigit():
            return 55   # Germany: 5-digit numeric
        if len(pc) == 4 and pc.isdigit() and city:
            return 161  # Norway

    # Infer from city name for well-known cities
    if city:
        city_lower = city.lower().strip()
        norwegian_cities = {"oslo", "bergen", "trondheim", "stavanger", "tromsø", "drammen",
                           "kristiansand", "fredrikstad", "sandnes", "bodø", "ålesund", "tønsberg",
                           "sandefjord", "haugesund", "arendal", "molde", "moss", "hamar",
                           "lillehammer", "gjøvik", "kongsberg", "halden", "larvik", "ski",
                           "mjøndalen", "lillestrøm", "jessheim", "ås", "narvik", "harstad"}
        if city_lower in norwegian_cities:
            return 161

    # No inference possible — don't guess
    return 161  # Still default Norway as most tasks are Norwegian context


def _get_logged_in_employee_id(tx: TripletexClient) -> int | None:
    """Get the logged-in employee ID. Cached."""
    if "logged_in_employee_id" in _cache:
        return _cache["logged_in_employee_id"]
    result = tx.get("token/session/>whoAmI")
    emp_id = result.get("value", {}).get("employeeId")
    _cache["logged_in_employee_id"] = emp_id
    return emp_id


def _ensure_department(tx: TripletexClient) -> int:
    """Get existing department ID, or create one if none exist. Cached."""
    if "department_id" in _cache:
        return _cache["department_id"]
    result = tx.get("department")
    departments = result.get("values", [])
    if departments:
        _cache["department_id"] = departments[0]["id"]
        return departments[0]["id"]
    result = tx.post("department", {"name": "Avdeling", "departmentNumber": "1"})
    dept_id = _get_id(result)
    if dept_id:
        _cache["department_id"] = dept_id
    return dept_id or 0


def _ensure_salary_settings(tx: TripletexClient, year: int) -> None:
    """Ensure salary module, settings, and standard time exist for payroll. Cached."""
    if _cache.get("salary_settings_ok"):
        return

    # Ensure WAGE module is active (required for salary transactions)
    modules = tx.get("company/salesmodules")
    has_wage = any(m.get("name") == "WAGE" for m in modules.get("values", []))
    if not has_wage:
        logger.info("Activating WAGE module for salary")
        tx.post("company/salesmodules", {"name": "WAGE"})

    # Check/update salary settings (payment day etc.)
    settings = tx.get("salary/settings")
    if not settings.get("_is_error"):
        val = settings.get("value", {})
        # Ensure payment day is set (default to 25th if missing)
        if not val.get("paymentDay"):
            tx.put("salary/settings", {
                "id": val.get("id"),
                "version": val.get("version", 0),
                "paymentDay": 25,
            })

    # Ensure standard time exists for the year
    std_time = tx.get("salary/settings/standardTime", {"from": 0, "count": 100})
    has_year = False
    for st in std_time.get("values", []):
        if st.get("year") == year:
            has_year = True
            break
    if not has_year:
        tx.post("salary/settings/standardTime", {
            "year": year,
            "hoursPerDay": 7.5,
        })
        logger.info("Created standard time for year %d", year)

    _cache["salary_settings_ok"] = True


def _ensure_bank_account(tx: TripletexClient) -> None:
    """Ensure account 1920 has a bank account number set. Cached."""
    if _cache.get("bank_account_set"):
        return
    result = tx.get("ledger/account", {"number": "1920"})
    accounts = result.get("values", [])
    if not accounts:
        _cache["bank_account_set"] = True
        return
    acct = accounts[0]
    if not acct.get("bankAccountNumber"):
        tx.put(f"ledger/account/{acct['id']}", {
            "id": acct["id"],
            "version": acct["version"],
            "bankAccountNumber": "12345678903",
        })
        logger.info("Set bank account number on account 1920")
    _cache["bank_account_set"] = True


def _find_entity(tx: TripletexClient, path: str, params: dict) -> dict | None:
    """Find an entity by search params. Returns first match or None."""
    result = tx.get(path, params)
    values = result.get("values", [])
    total = result.get("fullResultSize", 0)
    if total > len(values):
        logger.warning("Search %s returned %d/%d results — some may be missing", path, len(values), total)
    return values[0] if values else None


# ============================================================
# HIGH-LEVEL ACTIONS — Called by the LLM
# ============================================================

def create_employee(
    tx: TripletexClient,
    firstName: str,
    lastName: str,
    email: str,
    dateOfBirth: str | None = None,
    startDate: str | None = None,
    userType: str = "STANDARD",
    phoneNumberMobile: str | None = None,
    phoneNumberWork: str | None = None,
    phoneNumberHome: str | None = None,
    employeeNumber: str | None = None,
    nationalIdentityNumber: str | None = None,
    bankAccountNumber: str | None = None,
    addressLine1: str | None = None,
    postalCode: str | None = None,
    city: str | None = None,
    country: str | None = None,
    departmentName: str | None = None,
    # Employment details
    annualSalary: float | None = None,
    monthlySalary: float | None = None,
    hourlySalary: float | None = None,
    percentageOfFullTimeEquivalent: float | None = None,
    occupationCode: str | None = None,
    employmentForm: str | None = None,
    employmentType: str | None = None,
    remunerationType: str | None = None,
    workingHoursScheme: str | None = None,
    **kwargs,
) -> dict:
    """Create an employee with employment record and employment details.

    Handles: department lookup/creation, employee creation, employment, employment details.
    Employment details include salary, STYRK code, employment form, percentage, etc.
    """
    # Find or create department
    if departmentName:
        # Look for existing department by name
        depts = tx.get("department")
        dept_id = None
        for d in depts.get("values", []):
            if d.get("name", "").lower() == departmentName.lower():
                dept_id = d["id"]
                break
        if not dept_id:
            # Create the department
            dept_result = create_department(tx=tx, name=departmentName)
            dept_id = _get_id(dept_result) or _ensure_department(tx)
    else:
        dept_id = _ensure_department(tx)

    # Email is REQUIRED by Tripletex. If not provided, generate a valid one from the name.
    if not email or email.strip() == "":
        import re
        fn = re.sub(r'[^a-z0-9]', '', firstName.lower().replace('æ','ae').replace('ø','o').replace('å','aa').replace('é','e').replace('ü','u').replace('ö','o'))
        ln = re.sub(r'[^a-z0-9]', '', lastName.lower().replace('æ','ae').replace('ø','o').replace('å','aa').replace('é','e').replace('ü','u').replace('ö','o'))
        email = f"{fn}.{ln}@example.org"
        logger.info("No email provided — generated: %s", email)

    body: dict = {
        "firstName": firstName,
        "lastName": lastName,
        "email": email,
        "userType": userType,
        "department": {"id": dept_id},
    }
    if dateOfBirth:
        body["dateOfBirth"] = dateOfBirth
    if phoneNumberMobile:
        body["phoneNumberMobile"] = phoneNumberMobile
    if phoneNumberWork:
        body["phoneNumberWork"] = phoneNumberWork
    if phoneNumberHome:
        body["phoneNumberHome"] = phoneNumberHome
    if employeeNumber:
        body["employeeNumber"] = employeeNumber
    if nationalIdentityNumber:
        body["nationalIdentityNumber"] = nationalIdentityNumber
    if bankAccountNumber:
        body["bankAccountNumber"] = bankAccountNumber
    if addressLine1 or postalCode or city:
        addr: dict = {}
        if addressLine1:
            addr["addressLine1"] = addressLine1
        if postalCode:
            addr["postalCode"] = postalCode
        if city:
            addr["city"] = city
        addr["country"] = {"id": _get_country_id(country, city, postalCode)}
        body["address"] = addr

    _add_extra_fields(body, kwargs, "Employee")
    result = tx.post("employee", body)

    if result.get("_is_error"):
        return result

    emp_id = _get_id(result)
    if not emp_id:
        return result
    logger.info("Created employee %s %s (id=%d)", firstName, lastName, emp_id)

    # Create employment
    import datetime
    emp_start = startDate or datetime.date.today().isoformat()
    emp_body: dict = {"employee": {"id": emp_id}, "startDate": emp_start}
    emp_result = tx.post("employee/employment", emp_body)
    employment_id = _get_id(emp_result)
    if emp_result.get("_is_error"):
        logger.warning("Employment creation failed: %s", emp_result.get("_error_summary", ""))
    else:
        logger.info("Created employment for employee %d (employment_id=%s)", emp_id, employment_id)

    # Create employment details if any detail fields provided
    has_details = any([annualSalary, monthlySalary, hourlySalary, percentageOfFullTimeEquivalent,
                       occupationCode, employmentForm, employmentType, remunerationType, workingHoursScheme])
    if has_details and employment_id:
        details_body: dict = {
            "employment": {"id": employment_id},
            "date": emp_start,
        }

        # Salary
        if annualSalary:
            details_body["annualSalary"] = annualSalary
        if monthlySalary and not annualSalary:
            details_body["annualSalary"] = monthlySalary * 12
        if hourlySalary:
            details_body["hourlyWage"] = hourlySalary

        # Employment percentage
        if percentageOfFullTimeEquivalent is not None:
            details_body["percentageOfFullTimeEquivalent"] = percentageOfFullTimeEquivalent

        # STYRK occupation code
        if occupationCode:
            # Look up by code first (numeric), then by name
            occ = tx.get("employee/employment/occupationCode", {"code": str(occupationCode)})
            occ_vals = occ.get("values", [])
            if not occ_vals:
                occ = tx.get("employee/employment/occupationCode", {"nameNO": str(occupationCode)})
                occ_vals = occ.get("values", [])
            if not occ_vals:
                # Try broader search — fetch all and filter
                occ = tx.get("employee/employment/occupationCode", {"count": 5000})
                occ_vals = [o for o in occ.get("values", []) if str(o.get("code", "")) == str(occupationCode)]
            if occ_vals:
                details_body["occupationCode"] = {"id": occ_vals[0]["id"]}
                logger.info("Found STYRK code %s (id=%d)", occupationCode, occ_vals[0]["id"])
            else:
                logger.warning("STYRK code %s not found", occupationCode)

        # Employment form: PERMANENT, TEMPORARY, etc.
        form_map = {
            "fast": "PERMANENT", "permanent": "PERMANENT", "fast stilling": "PERMANENT",
            "befristet": "TEMPORARY", "temporary": "TEMPORARY", "midlertidig": "TEMPORARY",
            "vikariat": "TEMPORARY",
        }
        if employmentForm:
            details_body["employmentForm"] = form_map.get(employmentForm.lower(), employmentForm.upper())

        # Employment type
        if employmentType:
            details_body["employmentType"] = employmentType.upper()
        else:
            details_body["employmentType"] = "ORDINARY"

        # Remuneration type
        rem_map = {
            "fastlønn": "MONTHLY_WAGE", "månedslønn": "MONTHLY_WAGE", "månedlig": "MONTHLY_WAGE",
            "monthly": "MONTHLY_WAGE", "fastlonn": "MONTHLY_WAGE",
            "timelønn": "HOURLY_WAGE", "hourly": "HOURLY_WAGE",
            "provisjon": "COMMISION_PERCENTAGE", "commission": "COMMISION_PERCENTAGE",
            "fee": "FEE",
        }
        if remunerationType:
            details_body["remunerationType"] = rem_map.get(remunerationType.lower(), remunerationType.upper())
        elif annualSalary or monthlySalary:
            details_body["remunerationType"] = "MONTHLY_WAGE"
        elif hourlySalary:
            details_body["remunerationType"] = "HOURLY_WAGE"

        # Working hours scheme
        if workingHoursScheme:
            details_body["workingHoursScheme"] = workingHoursScheme.upper()
        else:
            details_body["workingHoursScheme"] = "NOT_SHIFT"

        details_result = tx.post("employee/employment/details", details_body)
        if details_result.get("_is_error"):
            logger.warning("Employment details failed: %s", details_result.get("_error_summary", ""))
        else:
            logger.info("Created employment details for employee %d (salary=%s, pct=%s, STYRK=%s)",
                        emp_id, annualSalary, percentageOfFullTimeEquivalent, occupationCode)

    return result


def create_customer(
    tx: TripletexClient,
    name: str,
    email: str | None = None,
    invoiceEmail: str | None = None,
    organizationNumber: str | None = None,
    phoneNumber: str | None = None,
    phoneNumberMobile: str | None = None,
    isPrivateIndividual: bool | None = None,
    language: str | None = None,
    addressLine1: str | None = None,
    postalCode: str | None = None,
    city: str | None = None,
    country: str | None = None,
    **kwargs,
) -> dict:
    """Create a customer with optional address."""
    country_id = _get_country_id(country, city, postalCode, entity_name=name, email=email, organizationNumber=organizationNumber)
    account_manager_id = _get_logged_in_employee_id(tx)

    body: dict = {
        "name": name,
        "isCustomer": True,
        "isSupplier": False,
        "isPrivateIndividual": False,
        "isInactive": False,
        "isFactoring": False,
        "invoicesDueIn": 14,
        "invoicesDueInType": "DAYS",
        "invoiceSendMethod": "EMAIL",
        "emailAttachmentType": "ATTACHMENT",
        "invoiceSendSMSNotification": False,
        "isAutomaticReminderEnabled": False,
        "isAutomaticSoftReminderEnabled": False,
        "isAutomaticNoticeOfDebtCollectionEnabled": False,
        "singleCustomerInvoice": False,
        "currency": {"id": 1},
        "language": "NO",
    }
    if account_manager_id:
        body["accountManager"] = {"id": account_manager_id}
    if email:
        body["email"] = email
        if not invoiceEmail:
            body["invoiceEmail"] = email
    if invoiceEmail:
        body["invoiceEmail"] = invoiceEmail
    if phoneNumberMobile:
        body["phoneNumberMobile"] = phoneNumberMobile
    if isPrivateIndividual is not None:
        body["isPrivateIndividual"] = isPrivateIndividual
    if language:
        body["language"] = language
    if organizationNumber:
        body["organizationNumber"] = organizationNumber
    if phoneNumber:
        body["phoneNumber"] = phoneNumber

    # Always set address with at least the country
    addr: dict = {"country": {"id": country_id}}
    if addressLine1:
        addr["addressLine1"] = addressLine1
    if postalCode:
        addr["postalCode"] = postalCode
    if city:
        addr["city"] = city
    body["physicalAddress"] = addr
    body["postalAddress"] = addr

    # Delivery address (separate from physical/postal)
    delivery_address = kwargs.pop("deliveryAddress", None)
    if delivery_address and isinstance(delivery_address, dict):
        del_addr: dict = {}
        if delivery_address.get("addressLine1"):
            del_addr["addressLine1"] = delivery_address["addressLine1"]
        if delivery_address.get("postalCode"):
            del_addr["postalCode"] = delivery_address["postalCode"]
        if delivery_address.get("city"):
            del_addr["city"] = delivery_address["city"]
        del_country = delivery_address.get("country")
        del_addr["country"] = {"id": _get_country_id(del_country)}
        body["deliveryAddress"] = del_addr

    _add_extra_fields(body, kwargs, "Customer")
    result = tx.post("customer", body)
    cid = _get_id(result)

    # Ensure address is set via PUT (some Tripletex proxies ignore address on POST)
    if cid and (addressLine1 or postalCode or city):
        cust = result.get("value", {})
        for addr_field in ["physicalAddress", "postalAddress"]:
            addr_obj = cust.get(addr_field, {})
            addr_id = addr_obj.get("id") if isinstance(addr_obj, dict) else None
            if addr_id:
                addr_body: dict = {"id": addr_id, "version": addr_obj.get("version", 0)}
                if addressLine1:
                    addr_body["addressLine1"] = addressLine1
                if postalCode:
                    addr_body["postalCode"] = postalCode
                if city:
                    addr_body["city"] = city
                addr_body["country"] = {"id": country_id}
                tx.put(f"address/{addr_id}", addr_body)

    logger.info("Created customer %s (id=%s)", name, cid)
    return result


def find_customer(
    tx: TripletexClient,
    name: str | None = None,
    organizationNumber: str | None = None,
    **kwargs,
) -> dict:
    """Find an existing customer by name or org number."""
    params = {}
    if organizationNumber:
        params["organizationNumber"] = organizationNumber
    elif name:
        params["name"] = name
    result = tx.get("customer", params)
    values = result.get("values", [])
    if values:
        logger.info("Found customer: %s (id=%d)", values[0].get("name"), values[0]["id"])
        return {"found": True, "customer": values[0]}

    return {"found": False, "message": f"Customer not found. Searched: name={name}, org={organizationNumber}"}


def create_product(
    tx: TripletexClient,
    name: str,
    priceExcludingVat: float,
    description: str | None = None,
    number: str | None = None,
    costPrice: float | None = None,
    vatTypeId: int | None = None,
    **kwargs,
) -> dict:
    """Create a product."""
    body: dict = {"name": name, "priceExcludingVatCurrency": priceExcludingVat}
    if description:
        body["description"] = description
    if number:
        body["number"] = number
    if costPrice is not None and costPrice > 0:
        body["costExcludingVatCurrency"] = costPrice
    if vatTypeId is not None:
        body["vatType"] = {"id": vatTypeId}

    # Product unit lookup by name
    product_unit = kwargs.pop("productUnit", None)
    if product_unit:
        units = tx.get("product/unit")
        unit_values = units.get("values", [])
        unit_id = None
        for u in unit_values:
            if (u.get("name", "").lower() == str(product_unit).lower()
                    or u.get("nameShort", "").lower() == str(product_unit).lower()):
                unit_id = u["id"]
                break
        if unit_id:
            body["productUnit"] = {"id": unit_id}
        else:
            logger.warning("Product unit '%s' not found among %d units", product_unit, len(unit_values))

    # Supplier reference
    supplier_id = kwargs.pop("supplierId", None)
    if supplier_id:
        body["supplier"] = {"id": int(supplier_id)}

    _add_extra_fields(body, kwargs, "Product")
    result = tx.post("product", body)
    logger.info("Created product %s (id=%s)", name, _get_id(result))
    return result


def create_invoice(
    tx: TripletexClient,
    customerId: int | None = None,
    customerName: str | None = None,
    organizationNumber: str | None = None,
    invoiceDate: str | None = None,
    invoiceDueDate: str | None = None,
    lines: list[dict] | None = None,
    projectId: int | None = None,
    **kwargs,
) -> dict:
    """Create an invoice.

    Lines: [{description, count, unitPrice, productId?, vatTypeId?, productNumber?, discount?}]
    - productId: Tripletex product ID (if known)
    - productNumber: Product number string to look up
    - vatTypeId: 3=25%, 31=15%, 32=12%, 5=0% exempt
    - count: quantity (default 1)
    - unitPrice: price per unit excluding VAT

    Handles: customer lookup, bank account setup, product lookup, order creation, invoice creation.
    """
    import datetime

    # Auto-resolve customer if name/org given instead of ID
    if not customerId and (customerName or organizationNumber):
        found = find_customer(tx=tx, name=customerName, organizationNumber=organizationNumber)
        if found.get("found"):
            customerId = found["customer"]["id"]
            logger.info("Auto-resolved customer '%s' to id=%d", customerName or organizationNumber, customerId)
        else:
            return {"_is_error": True, "_error_summary": f"Customer not found: name={customerName}, org={organizationNumber}. Create the customer first."}

    if not customerId:
        return {"_is_error": True, "_error_summary": "customerId is required. Use find_customer or create_customer first."}

    # Default dates
    if not invoiceDate:
        invoiceDate = datetime.date.today().isoformat()
    if not invoiceDueDate:
        d = datetime.date.fromisoformat(invoiceDate)
        invoiceDueDate = (d + datetime.timedelta(days=14)).isoformat()

    if not lines:
        return {"_is_error": True, "_error_summary": "Invoice lines are required. Provide at least one line with description and unitPrice."}

    _ensure_bank_account(tx)

    # VAT percentage to vatType ID mapping
    vat_map = {
        25: 3, 0.25: 3,
        15: 31, 0.15: 31,
        12: 32, 0.12: 32,
        0: 5, 0.0: 5,
    }

    # Create order lines with full support
    order_lines = []
    for line in lines:
        ol: dict = {
            "description": line.get("description", ""),
            "count": line.get("count", 1),
            "unitPriceExcludingVatCurrency": line.get("unitPrice", 0),
        }

        # Product reference
        product_id = line.get("productId") or None  # treat 0 as None
        product_number = line.get("productNumber")
        if product_number and not product_id:
            # Look up product by number
            result = tx.get("product", {"number": str(product_number)})
            products = result.get("values", [])
            if products:
                product_id = products[0]["id"]
                logger.info("Found product %s (id=%d)", product_number, product_id)
        if product_id:
            ol["product"] = {"id": product_id}

        # VAT type — vatTypeId 5 (0% exempt) should only be used when explicitly requested
        # "excl. VAT" / "ohne MwSt" means price is quoted without VAT, NOT 0% rate
        # Default is 25% (vatTypeId 3) — only override if prompt explicitly says 0%/15%/12%
        vat_type_id = line.get("vatTypeId")
        vat_pct = line.get("vatPercent") or line.get("vatRate") or line.get("vat")
        # Filter out 0% when it's likely a misinterpretation of "excl. VAT"
        if vat_type_id == 5 and (vat_pct == 0 or vat_pct is None):
            # Only keep 0% if the line description explicitly mentions exemption
            desc_lower = (line.get("description") or "").lower()
            if not any(kw in desc_lower for kw in ["avgiftsfri", "fritatt", "exempt", "exento", "0%",
                                                        "purregebyr", "purring", "reminder", "mahngebühr", "mahngebuhr",
                                                        "taxa de lembrete", "cargo por mora", "frais de rappel"]):
                vat_type_id = None  # Let Tripletex use default 25%
                vat_pct = None
                logger.info("Stripped vatTypeId=5 (0%% exempt) — likely misinterpreted 'excl. VAT'. Using default 25%%.")
        if vat_pct is not None and not vat_type_id:
            vat_type_id = vat_map.get(vat_pct) or vat_map.get(int(vat_pct)) if vat_pct else None
        if vat_type_id:
            ol["vatType"] = {"id": vat_type_id}

        # Discount — only include if non-zero
        discount = line.get("discount")
        if discount is not None and discount != 0 and discount != 0.0:
            ol["discount"] = discount

        order_lines.append(ol)

    # Currency handling — look up currency ID if specified on lines or kwargs
    currency_code = kwargs.get("currency")
    if not currency_code:
        for line in lines:
            if line.get("currency"):
                currency_code = line["currency"]
                break

    currency_id = 1  # NOK default
    if currency_code and currency_code.upper() != "NOK":
        currencies = tx.get("currency", {"code": currency_code.upper()})
        cur_values = currencies.get("values", [])
        if cur_values:
            currency_id = cur_values[0]["id"]
            logger.info("Using currency %s (id=%d)", currency_code, currency_id)

    # Get logged-in employee for ourContactEmployee
    contact_employee_id = _get_logged_in_employee_id(tx)

    order_body: dict = {
        "customer": {"id": customerId},
        "orderDate": invoiceDate,
        "deliveryDate": invoiceDate,
        "orderLines": order_lines,
        "currency": {"id": currency_id},
        "invoicesDueIn": 14,
        "invoicesDueInType": "DAYS",
    }
    if projectId:
        order_body["project"] = {"id": projectId}
    if contact_employee_id:
        order_body["ourContactEmployee"] = {"id": contact_employee_id}
    order_result = tx.post("order", order_body)

    if order_result.get("_is_error"):
        return order_result

    order_id = _get_id(order_result)
    if not order_id:
        return order_result
    logger.info("Created order %d", order_id)

    # Create invoice
    invoice_body = {
        "invoiceDate": invoiceDate,
        "invoiceDueDate": invoiceDueDate,
        "customer": {"id": customerId},
        "orders": [{"id": order_id}],
    }
    # Optional invoice fields
    kid = kwargs.pop("kid", None)
    if kid:
        invoice_body["kid"] = kid
    invoice_remark = kwargs.pop("invoiceRemark", None)
    if invoice_remark:
        invoice_body["invoiceRemarks"] = invoice_remark
    invoice_comment = kwargs.pop("invoiceComment", None)
    if invoice_comment:
        invoice_body["comment"] = invoice_comment
    result = tx.post("invoice", invoice_body)

    if result.get("_is_error"):
        return result

    logger.info("Created invoice %s", _get_id(result))
    return result


def find_invoice(
    tx: TripletexClient,
    customerId: int | None = None,
    invoiceNumber: str | None = None,
    **kwargs,
) -> dict:
    """Find invoices. Always includes required date params."""
    params = {
        "invoiceDateFrom": "2020-01-01",
        "invoiceDateTo": "2030-12-31",
    }
    if invoiceNumber:
        params["invoiceNumber"] = invoiceNumber
    if customerId and customerId > 0:
        params["customerId"] = customerId
    result = tx.get("invoice", params)
    values = result.get("values", [])
    logger.info("Found %d invoices", len(values))
    if not values:
        return {"found": False, "invoices": [], "message": f"No invoices found. Searched: customerId={customerId}, invoiceNumber={invoiceNumber}. Try broader search or check the customer exists."}
    return {"found": True, "invoices": values}


def create_credit_note(
    tx: TripletexClient,
    invoiceId: int,
    date: str | None = None,
    comment: str | None = None,
    **kwargs,
) -> dict:
    """Create a credit note for an invoice (reverses it).

    Uses query parameters as required by the /:createCreditNote endpoint.
    """
    import datetime
    if not date:
        date = datetime.date.today().isoformat()

    params: dict = {"date": date}
    if comment:
        params["comment"] = comment

    url = f"{tx.base_url}/invoice/{invoiceId}/:createCreditNote"
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}

    if resp.status_code >= 400:
        logger.warning("Credit note failed: %s", result)
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} creating credit note for invoice {invoiceId}"
    else:
        logger.info("Created credit note for invoice %d", invoiceId)

    return result


def create_project(
    tx: TripletexClient,
    name: str,
    projectManagerId: int | None = None,
    startDate: str | None = None,
    customerId: int | None = None,
    endDate: str | None = None,
    description: str | None = None,
    fixedprice: float | None = None,
    isInternal: bool | None = None,
    **kwargs,
) -> dict:
    """Create a project. Auto-resolves projectManagerId from logged-in user if not provided."""
    import datetime

    # Default project manager to logged-in employee
    if not projectManagerId:
        projectManagerId = _get_logged_in_employee_id(tx)
    if not projectManagerId:
        return {"_is_error": True, "_error_summary": "projectManagerId is required. Use find_employee first, or the logged-in employee will be used."}

    if not startDate:
        startDate = datetime.date.today().isoformat()

    body: dict = {
        "name": name,
        "projectManager": {"id": projectManagerId},
        "startDate": startDate,
    }
    # Only set customer if it's a real ID (not placeholder 0 or 1)
    if customerId and customerId > 100:
        body["customer"] = {"id": customerId}
    elif isInternal:
        pass  # Internal projects don't need a customer
    if endDate:
        body["endDate"] = endDate
    if description:
        body["description"] = description
    if fixedprice is not None:
        body["fixedprice"] = fixedprice
        body["isFixedPrice"] = True
    if isInternal is not None:
        body["isInternal"] = isInternal

    # Project category
    project_category_id = kwargs.pop("projectCategoryId", None)
    if project_category_id:
        body["projectCategory"] = {"id": int(project_category_id)}

    # Invoice comment on project
    invoice_comment = kwargs.pop("invoiceComment", None)
    if invoice_comment:
        body["invoiceComment"] = invoice_comment

    _add_extra_fields(body, kwargs, "Project")
    result = tx.post("project", body)
    logger.info("Created project %s (id=%s)", name, _get_id(result))
    return result


def find_project(
    tx: TripletexClient,
    name: str | None = None,
    **kwargs,
) -> dict:
    """Find a project by name."""
    params = {}
    if name:
        params["name"] = name
    result = tx.get("project", params)
    values = result.get("values", [])
    if values:
        return {"found": True, "project": values[0], "all_projects": values}
    return {"found": False, "message": f"No project found matching name='{name}'. Use create_project to create it."}


def find_employee(
    tx: TripletexClient,
    email: str | None = None,
    firstName: str | None = None,
    lastName: str | None = None,
    **kwargs,
) -> dict:
    """Find an employee by email or name. Single API call."""
    params = {}
    if email:
        params["email"] = email
    elif firstName or lastName:
        if firstName:
            params["firstName"] = firstName
        if lastName:
            params["lastName"] = lastName
    result = tx.get("employee", params)
    values = result.get("values", [])
    if values:
        logger.info("Found employee: %s (id=%d)", values[0].get("displayName"), values[0]["id"])
        return {"found": True, "employee": values[0]}

    return {"found": False, "message": f"Employee not found. Searched: email={email}, name={firstName} {lastName}"}


def create_travel_expense(
    tx: TripletexClient,
    employeeId: int,
    title: str,
    date: str,
    costs: list[dict] | None = None,
    perDiems: list[dict] | None = None,
    mileageAllowances: list[dict] | None = None,
    accommodationAllowances: list[dict] | None = None,
    projectId: int | None = None,
    departmentId: int | None = None,
    **kwargs,
) -> dict:
    """Create a travel expense with optional costs, per diems, and mileage.

    costs: [{description, amount, categoryKeyword}]
    perDiems: [{count (days), rate (NOK/day), overnightAccommodation?, location?, isDeductionForBreakfast/Lunch/Dinner?}]
    mileageAllowances: [{km, rate, date, departureLocation?, destination?, isCompanyCar?}]
    accommodationAllowances: [{count (nights), rate, location?}]
    """
    body: dict = {
        "employee": {"id": employeeId},
        "title": title,
    }
    if projectId:
        body["project"] = {"id": projectId}
    if departmentId:
        body["department"] = {"id": departmentId}

    # If per diems or mileage are included, this is a travel expense (reiseregning)
    # and needs travelDetails to distinguish from employee expense (ansattutlegg)
    if perDiems or mileageAllowances:
        import datetime
        departure = date
        return_date = date
        # Calculate return date from per diem days
        if perDiems:
            max_days = max(pd.get("count", 1) for pd in perDiems)
            if max_days > 1:
                try:
                    d = datetime.date.fromisoformat(date)
                    return_date = (d + datetime.timedelta(days=max_days - 1)).isoformat()
                except (ValueError, TypeError):
                    pass
        # Extract destination from per diem location, mileage, or title
        destination = kwargs.get("destination", "")
        departure_from = kwargs.get("departureFrom", "")
        if not destination and perDiems:
            for pd in perDiems:
                if pd.get("location"):
                    destination = pd["location"]
                    break
        if mileageAllowances:
            for ma in mileageAllowances:
                if not destination and ma.get("destination"):
                    destination = ma["destination"]
                if not departure_from and ma.get("departureLocation"):
                    departure_from = ma["departureLocation"]

        # Try extracting destination from title (e.g. "Client visit Trondheim" → "Trondheim")
        if not destination and title:
            # Common patterns: "Besøk/Visit/Visite/Visita/Konferanse/Conference + City"
            import re
            match = re.search(r'(?:visit|besøk|visite|visita|konferanse|conference|conferencia|conférence|kundenbesuch|kundebesøk)\s+(.+)', title, re.IGNORECASE)
            if match:
                destination = match.group(1).strip()

        # If no explicit departure, default to Oslo (company HQ assumption)
        if not departure_from:
            departure_from = "Oslo"

        body["travelDetails"] = {
            "departureDate": departure,
            "returnDate": return_date,
            "departureFrom": departure_from,
            "destination": destination,
            "purpose": title,
            "isDayTrip": not perDiems and not accommodationAllowances,
            "isForeignTravel": kwargs.get("isForeignTravel", False),
        }

    _add_extra_fields(body, kwargs, "TravelExpense")
    result = tx.post("travelExpense", body)

    if result.get("_is_error"):
        return result

    te_id = _get_id(result)
    if not te_id:
        return result
    logger.info("Created travel expense %d: %s", te_id, title)

    if costs:
        # Get available cost categories and payment types (cached)
        if "cost_categories" not in _cache:
            categories = tx.get("travelExpense/costCategory")
            _cache["cost_categories"] = categories.get("values", [])
        cat_values = _cache["cost_categories"]
        cat_map = {c["description"].lower(): c["id"] for c in cat_values}

        if "payment_type_id" not in _cache:
            payment_types = tx.get("travelExpense/paymentType")
            pt_values = payment_types.get("values", [])
            _cache["payment_type_id"] = pt_values[0]["id"] if pt_values else None
        payment_type_id = _cache["payment_type_id"]

        for cost in costs:
            # Find best matching category
            keyword = cost.get("categoryKeyword", "").lower()
            cat_id = None
            for cat_name, cid in cat_map.items():
                if keyword in cat_name or cat_name in keyword:
                    cat_id = cid
                    break
            if not cat_id:
                # Fallback to first active travel expense category
                for c in cat_values:
                    if c.get("showOnTravelExpenses"):
                        cat_id = c["id"]
                        break
            if not cat_id and cat_values:
                cat_id = cat_values[0]["id"]

            cost_body: dict = {
                "travelExpense": {"id": te_id},
                "date": date,
                "comments": cost.get("description", ""),
                "amountCurrencyIncVat": cost.get("amount", 0),
                "currency": {"id": 1},
            }
            if cat_id:
                cost_body["costCategory"] = {"id": cat_id}
            if payment_type_id:
                cost_body["paymentType"] = {"id": payment_type_id}

            cost_result = tx.post("travelExpense/cost", cost_body)
            if cost_result.get("_is_error"):
                logger.warning("Failed to add cost %s: %s", cost.get("description"), cost_result.get("_error_summary", ""))
            else:
                logger.info("Added cost: %s (%.0f NOK)", cost.get("description"), cost.get("amount", 0))

    # Add per diem compensations
    if perDiems:
        # Look up domestic per diem rate categories and rate types (cached)
        if "per_diem_overnight_rc" not in _cache:
            # Overnight category (multi-day trips)
            rc_overnight = tx.get("travelExpense/rateCategory", {
                "type": "PER_DIEM",
                "isValidDomestic": True,
                "isValidAccommodation": True,
            })
            rc_vals = [rc for rc in rc_overnight.get("values", [])
                       if not rc.get("toDate") or rc["toDate"] >= "2026-01-01"]
            overnight_rc_id = rc_vals[0]["id"] if rc_vals else None
            _cache["per_diem_overnight_rc"] = overnight_rc_id

            # Look up rate type for overnight
            if overnight_rc_id:
                rates = tx.get("travelExpense/rate", {"rateCategoryId": overnight_rc_id})
                rate_vals = rates.get("values", [])
                _cache["per_diem_overnight_rate"] = rate_vals[0]["id"] if rate_vals else None
            else:
                _cache["per_diem_overnight_rate"] = None

            # Day trip category
            rc_daytrip = tx.get("travelExpense/rateCategory", {
                "type": "PER_DIEM",
                "isValidDomestic": True,
                "isValidDayTrip": True,
            })
            rc_vals = [rc for rc in rc_daytrip.get("values", [])
                       if not rc.get("toDate") or rc["toDate"] >= "2026-01-01"]
            daytrip_rc_id = rc_vals[-1]["id"] if rc_vals else None
            _cache["per_diem_daytrip_rc"] = daytrip_rc_id

            if daytrip_rc_id:
                rates = tx.get("travelExpense/rate", {"rateCategoryId": daytrip_rc_id})
                rate_vals = rates.get("values", [])
                _cache["per_diem_daytrip_rate"] = rate_vals[0]["id"] if rate_vals else None
            else:
                _cache["per_diem_daytrip_rate"] = None

        for pd in perDiems:
            count = pd.get("count", 1)
            custom_rate = pd.get("rate")
            pd_body: dict = {
                "travelExpense": {"id": te_id},
                "count": count,
            }
            if pd.get("location"):
                pd_body["location"] = pd["location"]
                pd_body["address"] = pd["location"]

            # Set overnight accommodation — default to HOTEL for multi-day, NONE for day trips
            overnight = pd.get("overnightAccommodation")
            if overnight and overnight != "NONE":
                pd_body["overnightAccommodation"] = overnight
            elif count > 1:
                pd_body["overnightAccommodation"] = "HOTEL"
            else:
                pd_body["overnightAccommodation"] = "NONE"

            # Pick correct rate category and rate type: overnight for multi-day, day trip for single day
            is_overnight = pd_body.get("overnightAccommodation", "NONE") != "NONE"
            if is_overnight:
                if _cache.get("per_diem_overnight_rc"):
                    pd_body["rateCategory"] = {"id": _cache["per_diem_overnight_rc"]}
                if _cache.get("per_diem_overnight_rate"):
                    pd_body["rateType"] = {"id": _cache["per_diem_overnight_rate"]}
            else:
                if _cache.get("per_diem_daytrip_rc"):
                    pd_body["rateCategory"] = {"id": _cache["per_diem_daytrip_rc"]}
                if _cache.get("per_diem_daytrip_rate"):
                    pd_body["rateType"] = {"id": _cache["per_diem_daytrip_rate"]}

            # If a custom rate is specified, override rate and set amount
            # This must come AFTER rateCategory/rateType so the custom values take precedence
            if custom_rate:
                pd_body["rate"] = custom_rate
                pd_body["amount"] = custom_rate * count

            for meal in ["isDeductionForBreakfast", "isDeductionForLunch", "isDeductionForDinner"]:
                if pd.get(meal) is not None:
                    pd_body[meal] = pd[meal]

            logger.info("Per diem body: count=%d, rate=%s, amount=%s, overnight=%s",
                        count, pd_body.get("rate"), pd_body.get("amount"),
                        pd_body.get("overnightAccommodation"))
            pd_result = tx.post("travelExpense/perDiemCompensation", pd_body)
            if pd_result.get("_is_error"):
                logger.warning("Failed to add per diem: %s", pd_result.get("_error_summary", ""))
            else:
                # Log the actual rate/amount from the API response
                pd_val = pd_result.get("value", {})
                logger.info("Added per diem: %d days, rate=%s, amount=%s",
                            pd.get("count", 1), pd_val.get("rate"), pd_val.get("amount"))

    # Add mileage allowances
    if mileageAllowances:
        # Look up mileage rate category (cached)
        if "mileage_rate_category" not in _cache:
            rate_cats = tx.get("travelExpense/rateCategory", {"type": "MILEAGE_ALLOWANCE"})
            rc_values = rate_cats.get("values", [])
            if not rc_values:
                # Try without type filter
                rate_cats = tx.get("travelExpense/rateCategory")
                for rc in rate_cats.get("values", []):
                    if rc.get("type") == "MILEAGE_ALLOWANCE":
                        rc_values = [rc]
                        break
            _cache["mileage_rate_category"] = rc_values[0]["id"] if rc_values else None

        for ma in mileageAllowances:
            ma_body: dict = {
                "travelExpense": {"id": te_id},
                "km": ma.get("km", 0),
                "date": ma.get("date", date),
            }
            if ma.get("rate"):
                ma_body["rate"] = ma["rate"]
            if ma.get("departureLocation"):
                ma_body["departureLocation"] = ma["departureLocation"]
            if ma.get("destination"):
                ma_body["destination"] = ma["destination"]
            if ma.get("isCompanyCar") is not None:
                ma_body["isCompanyCar"] = ma["isCompanyCar"]
            if _cache.get("mileage_rate_category"):
                ma_body["rateCategory"] = {"id": _cache["mileage_rate_category"]}

            ma_result = tx.post("travelExpense/mileageAllowance", ma_body)
            if ma_result.get("_is_error"):
                logger.warning("Failed to add mileage: %s", ma_result.get("_error_summary", ""))
            else:
                logger.info("Added mileage: %.0f km", ma.get("km", 0))

    # Add accommodation allowances
    if accommodationAllowances:
        if "accommodation_rate_category" not in _cache:
            rate_cats = tx.get("travelExpense/rateCategory", {"type": "ACCOMMODATION_ALLOWANCE"})
            rc_values = rate_cats.get("values", [])
            _cache["accommodation_rate_category"] = rc_values[0]["id"] if rc_values else None

        for aa in accommodationAllowances:
            aa_body: dict = {
                "travelExpense": {"id": te_id},
                "count": aa.get("count", 1),
            }
            if aa.get("rate"):
                aa_body["rate"] = aa["rate"]
            if aa.get("location"):
                aa_body["location"] = aa["location"]
            if _cache.get("accommodation_rate_category"):
                aa_body["rateCategory"] = {"id": _cache["accommodation_rate_category"]}

            aa_result = tx.post("travelExpense/accommodationAllowance", aa_body)
            if aa_result.get("_is_error"):
                logger.warning("Failed to add accommodation: %s", aa_result.get("_error_summary", ""))
            else:
                logger.info("Added accommodation: %d nights", aa.get("count", 1))

    return result


def delete_travel_expense(
    tx: TripletexClient,
    employeeId: int | None = None,
    title: str | None = None,
    **kwargs,
) -> dict:
    """Find and delete a travel expense."""
    params = {}
    if employeeId:
        params["employeeId"] = employeeId
    result = tx.get("travelExpense", params)
    values = result.get("values", [])

    deleted = []
    for te in values:
        if title and title.lower() not in te.get("title", "").lower():
            continue
        tx.delete(f"travelExpense/{te['id']}")
        deleted.append(te["id"])
        logger.info("Deleted travel expense %d", te["id"])

    return {"deleted": deleted, "count": len(deleted)}


def register_timesheet(
    tx: TripletexClient,
    employeeId: int,
    projectId: int,
    date: str,
    hours: float,
    activityName: str | None = None,
    comment: str | None = None,
    **kwargs,
) -> dict:
    """Register timesheet hours. Auto-creates activity if none exists."""
    # Ensure employee is a participant on the project
    add_project_participant(tx=tx, projectId=projectId, employeeId=employeeId)

    # Find activity
    activities = tx.get("activity/>forTimeSheet", {
        "employeeId": employeeId,
        "projectId": projectId,
    })
    act_values = activities.get("values", [])

    activity_id = None
    if activityName and act_values:
        for a in act_values:
            if activityName.lower() in a.get("name", "").lower():
                activity_id = a["id"]
                break
    if not activity_id and act_values:
        activity_id = act_values[0]["id"]

    # Auto-create activity if none found
    if not activity_id:
        act_name = activityName or "Generell"
        logger.info("No activities found — creating '%s' for project %d", act_name, projectId)
        act_result = create_project_activity(tx=tx, name=act_name, projectId=projectId)
        act_id = _get_id(act_result)
        if act_id:
            activity_id = act_id
        else:
            # Try fetching again — activity might have been created globally
            activities = tx.get("activity/>forTimeSheet", {
                "employeeId": employeeId,
                "projectId": projectId,
            })
            act_values = activities.get("values", [])
            if act_values:
                activity_id = act_values[0]["id"]

    if not activity_id:
        return {"_is_error": True, "_error_summary": "No activities available for this project. Could not auto-create one."}

    body: dict = {
        "employee": {"id": employeeId},
        "project": {"id": projectId},
        "activity": {"id": activity_id},
        "date": date,
        "hours": hours,
    }
    if comment:
        body["comment"] = comment

    # Optional timesheet fields
    hourly_rate = kwargs.pop("hourlyRate", None)
    if hourly_rate is not None and hourly_rate > 0:
        body["hourlyRate"] = hourly_rate
    chargeable = kwargs.pop("chargeable", None)
    if chargeable is not None:
        body["chargeable"] = chargeable

    result = tx.post("timesheet/entry", body)
    if not result.get("_is_error"):
        logger.info("Registered %.1f hours for employee %d on project %d", hours, employeeId, projectId)
    return result


def create_department(
    tx: TripletexClient,
    name: str,
    departmentNumber: str | None = None,
    **kwargs,
) -> dict:
    """Create a department."""
    body: dict = {"name": name}
    if departmentNumber:
        body["departmentNumber"] = departmentNumber

    # Department manager reference — must be an EMPLOYEE ID, not a department ID
    # Employee IDs are typically > 10000000 on Tripletex
    dept_manager_id = kwargs.pop("departmentManagerId", None)
    if dept_manager_id and int(dept_manager_id) > 10000000:
        body["departmentManager"] = {"id": int(dept_manager_id)}

    _add_extra_fields(body, kwargs, "Department")
    result = tx.post("department", body)
    logger.info("Created department %s (id=%s)", name, _get_id(result))
    return result


def create_supplier(
    tx: TripletexClient,
    name: str,
    email: str | None = None,
    invoiceEmail: str | None = None,
    organizationNumber: str | None = None,
    phoneNumber: str | None = None,
    phoneNumberMobile: str | None = None,
    isPrivateIndividual: bool | None = None,
    language: str | None = None,
    addressLine1: str | None = None,
    postalCode: str | None = None,
    city: str | None = None,
    country: str | None = None,
    **kwargs,
) -> dict:
    """Create a supplier with optional address."""
    country_id = _get_country_id(country, city, postalCode, entity_name=name, email=email, organizationNumber=organizationNumber)
    account_manager_id = _get_logged_in_employee_id(tx)

    body: dict = {
        "name": name,
        "isSupplier": True,
        "isCustomer": False,
        "isPrivateIndividual": isPrivateIndividual if isPrivateIndividual is not None else False,
        "isInactive": False,
        "showProducts": False,
        "currency": {"id": 1},
        "language": "NO",
    }
    if account_manager_id:
        body["accountManager"] = {"id": account_manager_id}
    if email:
        body["email"] = email
        if not invoiceEmail:
            body["invoiceEmail"] = email
    if invoiceEmail:
        body["invoiceEmail"] = invoiceEmail
    if organizationNumber:
        body["organizationNumber"] = organizationNumber
    if phoneNumber:
        body["phoneNumber"] = phoneNumber
    if phoneNumberMobile:
        body["phoneNumberMobile"] = phoneNumberMobile
    if language:
        body["language"] = language

    # Always set address with at least the country
    addr: dict = {"country": {"id": country_id}}
    if addressLine1:
        addr["addressLine1"] = addressLine1
    if postalCode:
        addr["postalCode"] = postalCode
    if city:
        addr["city"] = city
    body["physicalAddress"] = addr
    body["postalAddress"] = addr

    # Delivery address (separate from physical/postal)
    delivery_address = kwargs.pop("deliveryAddress", None)
    if delivery_address and isinstance(delivery_address, dict):
        del_addr: dict = {}
        if delivery_address.get("addressLine1"):
            del_addr["addressLine1"] = delivery_address["addressLine1"]
        if delivery_address.get("postalCode"):
            del_addr["postalCode"] = delivery_address["postalCode"]
        if delivery_address.get("city"):
            del_addr["city"] = delivery_address["city"]
        del_country = delivery_address.get("country")
        del_addr["country"] = {"id": _get_country_id(del_country)}
        body["deliveryAddress"] = del_addr

    _add_extra_fields(body, kwargs, "Supplier")
    result = tx.post("supplier", body)
    sid = _get_id(result)

    # Ensure address is set via PUT (some Tripletex proxies ignore address on POST)
    if sid and (addressLine1 or postalCode or city):
        sup = result.get("value", {})
        for addr_field in ["physicalAddress", "postalAddress"]:
            addr_obj = sup.get(addr_field, {})
            addr_id = addr_obj.get("id") if isinstance(addr_obj, dict) else None
            if addr_id:
                addr_body: dict = {"id": addr_id, "version": addr_obj.get("version", 0)}
                if addressLine1:
                    addr_body["addressLine1"] = addressLine1
                if postalCode:
                    addr_body["postalCode"] = postalCode
                if city:
                    addr_body["city"] = city
                addr_body["country"] = {"id": country_id}
                tx.put(f"address/{addr_id}", addr_body)

    logger.info("Created supplier %s (id=%s)", name, sid)
    return result


def create_contact(
    tx: TripletexClient,
    firstName: str,
    lastName: str,
    customerId: int,
    email: str | None = None,
    phoneNumberMobile: str | None = None,
    phoneNumberWork: str | None = None,
    **kwargs,
) -> dict:
    """Create a contact on a customer."""
    body: dict = {
        "firstName": firstName,
        "lastName": lastName,
        "customer": {"id": customerId},
    }
    if email:
        body["email"] = email
    if phoneNumberMobile:
        body["phoneNumberMobile"] = phoneNumberMobile
    if phoneNumberWork:
        body["phoneNumberWork"] = phoneNumberWork
    _add_extra_fields(body, kwargs, "Contact")
    result = tx.post("contact", body)
    logger.info("Created contact %s %s (id=%s)", firstName, lastName, _get_id(result))
    return result


def update_employee(
    tx: TripletexClient,
    employeeId: int,
    firstName: str | None = None,
    lastName: str | None = None,
    email: str | None = None,
    dateOfBirth: str | None = None,
    phoneNumberMobile: str | None = None,
    phoneNumberWork: str | None = None,
    phoneNumberHome: str | None = None,
    **kwargs,
) -> dict:
    """Update an existing employee's fields."""
    # Get current version
    current = tx.get(f"employee/{employeeId}")
    emp = current.get("value", {})
    body = {"id": employeeId, "version": emp.get("version", 0)}

    if firstName:
        body["firstName"] = firstName
    if lastName:
        body["lastName"] = lastName
    if email:
        body["email"] = email
    if dateOfBirth:
        body["dateOfBirth"] = dateOfBirth
    if phoneNumberMobile:
        body["phoneNumberMobile"] = phoneNumberMobile
    if phoneNumberWork:
        body["phoneNumberWork"] = phoneNumberWork
    if phoneNumberHome:
        body["phoneNumberHome"] = phoneNumberHome

    result = tx.put(f"employee/{employeeId}", body)
    logger.info("Updated employee %d", employeeId)
    return result


def update_customer(
    tx: TripletexClient,
    customerId: int,
    name: str | None = None,
    email: str | None = None,
    phoneNumber: str | None = None,
    organizationNumber: str | None = None,
    addressLine1: str | None = None,
    postalCode: str | None = None,
    city: str | None = None,
    country: str | None = None,
    **kwargs,
) -> dict:
    """Update an existing customer's fields."""
    current = tx.get(f"customer/{customerId}")
    cust = current.get("value", {})
    body: dict = {"id": customerId, "version": cust.get("version", 0)}

    if name:
        body["name"] = name
    if email:
        body["email"] = email
    if phoneNumber:
        body["phoneNumber"] = phoneNumber
    if organizationNumber:
        body["organizationNumber"] = organizationNumber

    result = tx.put(f"customer/{customerId}", body)
    if result.get("_is_error"):
        logger.warning("Update customer failed: %s", result.get("_error_summary", ""))
        return result

    # Update address via direct PUT on address objects
    if addressLine1 or postalCode or city or country:
        cust_name = cust.get("name", "")
        updated_cust = result.get("value", cust)
        for addr_field in ["physicalAddress", "postalAddress"]:
            addr_obj = updated_cust.get(addr_field, {})
            addr_id = addr_obj.get("id") if isinstance(addr_obj, dict) else None
            if addr_id:
                addr_body: dict = {"id": addr_id, "version": addr_obj.get("version", 0)}
                if addressLine1:
                    addr_body["addressLine1"] = addressLine1
                if postalCode:
                    addr_body["postalCode"] = postalCode
                if city:
                    addr_body["city"] = city
                addr_body["country"] = {"id": _get_country_id(country, city, postalCode, entity_name=cust_name)}
                tx.put(f"address/{addr_id}", addr_body)
                logger.info("Updated %s on customer %d", addr_field, customerId)

    logger.info("Updated customer %d", customerId)
    return result


def book_exchange_difference(
    tx: TripletexClient,
    invoiceId: int,
    paymentAmountNOK: float,
    originalAmountNOK: float | None = None,
    paymentDate: str | None = None,
    exchangeAccount: int = 7790,
    **kwargs,
) -> dict:
    """Register payment on a foreign currency invoice AND book the exchange difference (agio/disagio).

    Handles the full flow:
    1. Gets the invoice to find the original NOK amount
    2. Registers the payment with the invoice outstanding amount
    3. Calculates the exchange difference (original vs received)
    4. Creates a voucher for the difference on the exchange account (default 7790)

    paymentAmountNOK: The actual NOK amount received/paid (foreign amount × current rate)
    originalAmountNOK: The original NOK amount when invoice was created (foreign amount × original rate).
                       If not provided, uses the invoice's outstanding amount.
    exchangeAccount: Account for exchange gains/losses (default 7790 Valutadifferanse)
    """
    import datetime
    if not paymentDate:
        paymentDate = datetime.date.today().isoformat()

    # 1. Get the invoice to find original amount
    invoice = tx.get(f"invoice/{invoiceId}")
    inv = invoice.get("value", {})
    invoice_amount = inv.get("amount", 0)
    outstanding = inv.get("amountOutstanding", invoice_amount)
    customer_id = None
    cust = inv.get("customer", {})
    if isinstance(cust, dict):
        customer_id = cust.get("id")

    # Use explicitly provided originalAmountNOK, or fall back to invoice outstanding
    if originalAmountNOK and originalAmountNOK > 0:
        original_amount = originalAmountNOK
    else:
        original_amount = outstanding

    logger.info("Exchange diff: invoice %d, invoice_amount=%s, outstanding=%s, original=%s, payment=%s",
                invoiceId, invoice_amount, outstanding, original_amount, paymentAmountNOK)

    # 2. Register payment with the OUTSTANDING amount (closes the invoice)
    pay_result = register_payment(tx=tx, invoiceId=invoiceId, paymentDate=paymentDate)
    if pay_result.get("_is_error"):
        return pay_result

    # 3. Calculate exchange difference using the original NOK amount
    # If paymentAmountNOK > original: agio (gain) — we received more
    # If paymentAmountNOK < original: disagio (loss) — we received less
    diff = round(paymentAmountNOK - original_amount, 2)

    if abs(diff) < 0.01:
        logger.info("No exchange difference (diff=%.2f)", diff)
        return {"payment": pay_result, "exchange_difference": 0, "message": "No exchange difference"}

    # 4. Create voucher for exchange difference
    postings = []
    if diff > 0:
        # Agio (gain): we received MORE than invoice amount
        # Debit bank (1920), Credit exchange account (7790)
        postings = [
            {"accountNumber": 1920, "debitAmount": diff, "creditAmount": 0,
             "description": f"Valutagevinst (agio) faktura {invoiceId}"},
            {"accountNumber": exchangeAccount, "debitAmount": 0, "creditAmount": diff,
             "description": f"Agio faktura {invoiceId}"},
        ]
    else:
        # Disagio (loss): we received LESS than invoice amount
        # Debit exchange account (7790), Credit bank (1920)
        loss = abs(diff)
        postings = [
            {"accountNumber": exchangeAccount, "debitAmount": loss, "creditAmount": 0,
             "description": f"Valutatap (disagio) faktura {invoiceId}"},
            {"accountNumber": 1920, "debitAmount": 0, "creditAmount": loss,
             "description": f"Disagio faktura {invoiceId}"},
        ]

    voucher_result = create_voucher(
        tx=tx, date=paymentDate,
        description=f"Valutadifferanse faktura {invoiceId}: {'agio' if diff > 0 else 'disagio'} {abs(diff):.2f} NOK",
        postings=postings,
    )

    return {
        "payment": pay_result,
        "exchange_difference": diff,
        "type": "agio" if diff > 0 else "disagio",
        "voucher": voucher_result,
    }


def register_payment(
    tx: TripletexClient,
    invoiceId: int,
    paymentDate: str | None = None,
    paymentAmount: float | None = None,
    **kwargs,
) -> dict:
    """Register a payment on an invoice.

    Uses query parameters (not body) as required by the /:payment endpoint.
    paymentDate defaults to today if not provided.
    paymentAmount defaults to the full outstanding amount if not provided.
    """
    import datetime
    if not paymentDate:
        paymentDate = datetime.date.today().isoformat()

    # Get payment types (cached)
    if "invoice_payment_type_id" not in _cache:
        payment_types = tx.get("invoice/paymentType")
        pt_values = payment_types.get("values", [])
        _cache["invoice_payment_type_id"] = pt_values[0]["id"] if pt_values else 0
    payment_type_id = _cache["invoice_payment_type_id"]

    # Allow override of payment type
    payment_type_override = kwargs.pop("paymentTypeId", None)
    if payment_type_override is not None and int(payment_type_override) > 0:
        payment_type_id = int(payment_type_override)

    # Always get the invoice to find the correct outstanding amount
    # This ensures we pay exactly what's owed, not a calculated guess
    invoice = tx.get(f"invoice/{invoiceId}")
    inv = invoice.get("value", {})
    invoice_amount = inv.get("amount", 0)
    outstanding = inv.get("amountOutstanding", invoice_amount)

    if paymentAmount is None or paymentAmount == 0:
        paymentAmount = outstanding
        logger.info("Auto-detected payment amount from invoice: %.2f", paymentAmount)
    else:
        # LLM provided an amount — but use invoice amount for full payment
        # If the LLM amount is close to the invoice amount, use the exact invoice amount
        if outstanding > 0 and abs(paymentAmount - outstanding) / outstanding < 0.01:
            paymentAmount = outstanding
            logger.info("Corrected payment to exact invoice outstanding: %.2f", paymentAmount)
        else:
            logger.info("Using LLM-provided payment amount: %.2f (invoice outstanding: %.2f)", paymentAmount, outstanding)

    # /:payment endpoint takes query params, NOT body
    params = {
        "paymentDate": paymentDate,
        "paymentTypeId": payment_type_id,
        "paidAmount": paymentAmount,
    }

    url = f"{tx.base_url}/invoice/{invoiceId}/:payment"
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}

    if resp.status_code >= 400:
        logger.warning("Payment failed: %s", result)
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} registering payment on invoice {invoiceId}"
    else:
        logger.info("Registered payment on invoice %d (%.2f NOK)", invoiceId, paymentAmount)

    return result


def reverse_payment(
    tx: TripletexClient,
    invoiceId: int,
    **kwargs,
) -> dict:
    """Reverse/undo a payment on an invoice (e.g. bank return).

    Finds the payment voucher on the invoice and reverses it,
    so the invoice shows the outstanding amount again.
    """
    # Get the invoice with postings to find the payment voucher
    invoice = tx.get(f"invoice/{invoiceId}")
    inv = invoice.get("value", {})

    # Get all postings for this invoice — payment postings have a different voucher than the invoice voucher
    invoice_voucher_id = None
    inv_voucher = inv.get("voucher")
    if isinstance(inv_voucher, dict):
        invoice_voucher_id = inv_voucher.get("id")

    # Find postings on account 1920 (bank) or similar — these are payment postings
    postings = tx.get("ledger/posting", {
        "invoiceId": invoiceId,
        "dateFrom": "2020-01-01",
        "dateTo": "2030-12-31",
    })
    posting_values = postings.get("values", [])

    # Find voucher IDs that are NOT the invoice voucher (those are payment vouchers)
    payment_voucher_ids = set()
    for p in posting_values:
        v = p.get("voucher", {})
        vid = v.get("id") if isinstance(v, dict) else None
        if vid and vid != invoice_voucher_id:
            payment_voucher_ids.add(vid)

    if not payment_voucher_ids:
        # Try alternative: look for postings on bank account (1920)
        bank_postings = tx.get("ledger/posting", {
            "customerId": inv.get("customer", {}).get("id"),
            "dateFrom": "2020-01-01",
            "dateTo": "2030-12-31",
        })
        for p in bank_postings.get("values", []):
            acct = p.get("account", {})
            acct_num = acct.get("number", 0)
            v = p.get("voucher", {})
            vid = v.get("id") if isinstance(v, dict) else None
            if vid and vid != invoice_voucher_id and acct_num in (1920, 1900, 1500):
                payment_voucher_ids.add(vid)

    if not payment_voucher_ids:
        return {"_is_error": True, "_error_summary": f"No payment voucher found for invoice {invoiceId}"}

    # Reverse all payment vouchers
    reversed_vouchers = []
    for vid in payment_voucher_ids:
        result = reverse_voucher(tx=tx, voucherId=vid)
        if result.get("_is_error"):
            logger.warning("Failed to reverse payment voucher %d: %s", vid, result.get("_error_summary", ""))
        else:
            reversed_vouchers.append(vid)
            logger.info("Reversed payment voucher %d for invoice %d", vid, invoiceId)

    return {"reversed_vouchers": reversed_vouchers, "invoice_id": invoiceId}


def grant_admin_role(
    tx: TripletexClient,
    employeeId: int,
    **kwargs,
) -> dict:
    """Grant administrator entitlements to an employee.

    Uses query parameters as required by the :grantEntitlementsByTemplate endpoint.
    """
    url = f"{tx.base_url}/employee/entitlement/:grantEntitlementsByTemplate"
    params = {"employeeId": employeeId, "template": "all"}
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}

    if resp.status_code >= 400:
        logger.warning("Grant admin failed: %s", result)
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} granting admin to employee {employeeId}"
    else:
        logger.info("Granted admin entitlements to employee %d", employeeId)

    return result


def enable_department_accounting(
    tx: TripletexClient,
    **kwargs,
) -> dict:
    """Enable the department accounting module by creating a department."""
    # Check if departments already exist
    result = tx.get("department")
    departments = result.get("values", [])
    if departments:
        logger.info("Department accounting already enabled (found %d departments)", len(departments))
        return {"status": "already_enabled", "departments": departments}

    # Creating a department implicitly enables department accounting
    dept_result = tx.post("department", {"name": "Avdeling 1", "departmentNumber": "1"})
    if dept_result.get("_is_error"):
        logger.warning("Enable department accounting failed: %s", dept_result.get("_error_summary", ""))
    else:
        logger.info("Enabled department accounting by creating first department")
    return dept_result


def delete_entity(
    tx: TripletexClient,
    entityType: str,
    entityId: int,
    **kwargs,
) -> dict:
    """Delete any entity by type and ID."""
    path = {
        "employee": "employee",
        "customer": "customer",
        "product": "product",
        "invoice": "invoice",
        "order": "order",
        "travelExpense": "travelExpense",
        "project": "project",
        "department": "department",
        "supplier": "supplier",
        "contact": "contact",
        "voucher": "ledger/voucher",
    }.get(entityType, entityType)

    result = tx.delete(f"{path}/{entityId}")
    logger.info("Deleted %s %d", entityType, entityId)
    return result


def find_entity(
    tx: TripletexClient,
    entityType: str,
    searchParams: dict | None = None,
    **kwargs,
) -> dict:
    """Find any entity by type and search params."""
    path = {
        "employee": "employee",
        "customer": "customer",
        "product": "product",
        "invoice": "invoice",
        "order": "order",
        "travelExpense": "travelExpense",
        "project": "project",
        "department": "department",
        "supplier": "supplier",
        "contact": "contact",
        "voucher": "ledger/voucher",
        "supplierInvoice": "supplierInvoice",
        "incomingInvoice": "incomingInvoice",
        "bankStatement": "bank/statement",
    }.get(entityType, entityType)

    params = searchParams or {}
    if path == "invoice":
        params.setdefault("invoiceDateFrom", "2020-01-01")
        params.setdefault("invoiceDateTo", "2030-12-31")

    result = tx.get(path, params)
    values = result.get("values", [])
    if not values:
        return {"found": False, "count": 0, "values": [], "message": f"No {entityType} found matching {searchParams or 'no filters'}. Check entity type and search params."}
    return {"found": True, "count": len(values), "values": values}


def update_product(
    tx: TripletexClient,
    productId: int,
    name: str | None = None,
    priceExcludingVat: float | None = None,
    description: str | None = None,
    **kwargs,
) -> dict:
    """Update an existing product."""
    current = tx.get(f"product/{productId}")
    prod = current.get("value", {})
    if not prod:
        return current
    body: dict = {"id": productId, "version": prod.get("version", 0)}
    if name is not None:
        body["name"] = name
    if priceExcludingVat is not None:
        body["priceExcludingVatCurrency"] = priceExcludingVat
    if description is not None:
        body["description"] = description
    result = tx.put(f"product/{productId}", body)
    if result.get("_is_error"):
        logger.warning("Update product failed: %s", result.get("_error_summary", ""))
    else:
        logger.info("Updated product %d", productId)
    return result


def update_supplier(
    tx: TripletexClient,
    supplierId: int,
    name: str | None = None,
    email: str | None = None,
    phoneNumber: str | None = None,
    organizationNumber: str | None = None,
    addressLine1: str | None = None,
    postalCode: str | None = None,
    city: str | None = None,
    country: str | None = None,
    **kwargs,
) -> dict:
    """Update an existing supplier."""
    current = tx.get(f"supplier/{supplierId}")
    sup = current.get("value", {})
    if not sup:
        return current
    body: dict = {"id": supplierId, "version": sup.get("version", 0)}
    if name is not None:
        body["name"] = name
    if email is not None:
        body["email"] = email
    if phoneNumber is not None:
        body["phoneNumber"] = phoneNumber
    if organizationNumber is not None:
        body["organizationNumber"] = organizationNumber
    result = tx.put(f"supplier/{supplierId}", body)
    if result.get("_is_error"):
        logger.warning("Update supplier failed: %s", result.get("_error_summary", ""))
        return result

    # Update address via direct PUT on address objects
    if addressLine1 or postalCode or city or country:
        sup_name = sup.get("name", "")
        updated_sup = result.get("value", sup)
        for addr_field in ["physicalAddress", "postalAddress"]:
            addr_obj = updated_sup.get(addr_field, {})
            addr_id = addr_obj.get("id") if isinstance(addr_obj, dict) else None
            if addr_id:
                addr_body: dict = {"id": addr_id, "version": addr_obj.get("version", 0)}
                if addressLine1:
                    addr_body["addressLine1"] = addressLine1
                if postalCode:
                    addr_body["postalCode"] = postalCode
                if city:
                    addr_body["city"] = city
                addr_body["country"] = {"id": _get_country_id(country, city, postalCode, entity_name=sup_name)}
                tx.put(f"address/{addr_id}", addr_body)
                logger.info("Updated %s on supplier %d", addr_field, supplierId)

    logger.info("Updated supplier %d", supplierId)
    return result


def update_contact(
    tx: TripletexClient,
    contactId: int,
    firstName: str | None = None,
    lastName: str | None = None,
    email: str | None = None,
    phoneNumberMobile: str | None = None,
    phoneNumberWork: str | None = None,
    **kwargs,
) -> dict:
    """Update an existing contact."""
    current = tx.get(f"contact/{contactId}")
    contact = current.get("value", {})
    if not contact:
        return current
    body: dict = {"id": contactId, "version": contact.get("version", 0)}
    if firstName is not None:
        body["firstName"] = firstName
    if lastName is not None:
        body["lastName"] = lastName
    if email is not None:
        body["email"] = email
    if phoneNumberMobile is not None:
        body["phoneNumberMobile"] = phoneNumberMobile
    if phoneNumberWork is not None:
        body["phoneNumberWork"] = phoneNumberWork
    result = tx.put(f"contact/{contactId}", body)
    if result.get("_is_error"):
        logger.warning("Update contact failed: %s", result.get("_error_summary", ""))
    else:
        logger.info("Updated contact %d", contactId)
    return result


def update_department(
    tx: TripletexClient,
    departmentId: int,
    name: str | None = None,
    departmentNumber: str | None = None,
    **kwargs,
) -> dict:
    """Update an existing department."""
    current = tx.get(f"department/{departmentId}")
    dept = current.get("value", {})
    if not dept:
        return current
    body: dict = {"id": departmentId, "version": dept.get("version", 0)}
    if name is not None:
        body["name"] = name
    if departmentNumber is not None:
        body["departmentNumber"] = departmentNumber
    result = tx.put(f"department/{departmentId}", body)
    if result.get("_is_error"):
        logger.warning("Update department failed: %s", result.get("_error_summary", ""))
    else:
        logger.info("Updated department %d", departmentId)
    return result


def update_travel_expense(
    tx: TripletexClient,
    travelExpenseId: int,
    title: str | None = None,
    date: str | None = None,
    **kwargs,
) -> dict:
    """Update an existing travel expense."""
    current = tx.get(f"travelExpense/{travelExpenseId}")
    te = current.get("value", {})
    body: dict = {"id": travelExpenseId, "version": te.get("version", 0)}
    if title:
        body["title"] = title
    if date:
        body["date"] = date
    result = tx.put(f"travelExpense/{travelExpenseId}", body)
    logger.info("Updated travel expense %d", travelExpenseId)
    return result


def update_project(
    tx: TripletexClient,
    projectId: int,
    name: str | None = None,
    fixedprice: float | None = None,
    isFixedPrice: bool | None = None,
    startDate: str | None = None,
    endDate: str | None = None,
    description: str | None = None,
    **kwargs,
) -> dict:
    """Update an existing project's fields (e.g. set fixed price)."""
    current = tx.get(f"project/{projectId}")
    proj = current.get("value", {})
    if not proj:
        return current

    body: dict = {"id": projectId, "version": proj.get("version", 0)}
    # Only include fields with actual values — empty strings can overwrite existing data
    pm_id = kwargs.pop("projectManagerId", None)
    if pm_id and int(pm_id) > 0:
        body["projectManager"] = {"id": int(pm_id)}
    if name and name.strip():
        body["name"] = name
    if fixedprice is not None and fixedprice != 0:
        body["fixedprice"] = fixedprice
        body["isFixedPrice"] = True
    if isFixedPrice is not None:
        body["isFixedPrice"] = isFixedPrice
    if startDate and startDate.strip():
        body["startDate"] = startDate
    if endDate and endDate.strip():
        body["endDate"] = endDate
    if description and description.strip():
        body["description"] = description

    result = tx.put(f"project/{projectId}", body)
    if result.get("_is_error"):
        logger.warning("Update project failed: %s", result.get("_error_summary", ""))
    else:
        logger.info("Updated project %d", projectId)
    return result


def find_accounts(
    tx: TripletexClient,
    number: int | str | None = None,
    name: str | None = None,
    **kwargs,
) -> dict:
    """Query the chart of accounts (ledger/account)."""
    params: dict = {}
    if number is not None:
        params["number"] = str(number)
    if name:
        params["name"] = name
    result = tx.get("ledger/account", params)
    values = result.get("values", [])
    logger.info("Found %d accounts", len(values))
    if not values:
        return {"found": False, "count": 0, "accounts": [], "message": f"No accounts found matching number={number}, name={name}."}
    return {"found": True, "count": len(values), "accounts": values}


def find_postings(
    tx: TripletexClient,
    dateFrom: str = "2020-01-01",
    dateTo: str = "2030-12-31",
    accountId: int | None = None,
    accountNumber: int | None = None,
    supplierId: int | None = None,
    customerId: int | None = None,
    **kwargs,
) -> dict:
    """Query ledger postings.

    accountId: Tripletex internal account ID
    accountNumber: Account number (e.g. 6300) — auto-resolves to accountId
    If the LLM passes an accountId that looks like an account number (>= 1000),
    we treat it as accountNumber and resolve it.
    """
    # Auto-detect: if accountId looks like an account number (>= 1000), resolve it
    if accountId and accountId >= 1000 and not accountNumber:
        accountNumber = accountId
        accountId = None

    # Resolve accountNumber to accountId
    if accountNumber and not accountId:
        if "account_map" not in _cache:
            _cache["account_map"] = {}
        acct_id = _cache["account_map"].get(accountNumber)
        if not acct_id:
            lookup = tx.get("ledger/account", {"number": str(accountNumber)})
            vals = lookup.get("values", [])
            if vals:
                acct_id = vals[0]["id"]
                _cache["account_map"][accountNumber] = acct_id
                logger.info("Resolved account number %d to id %d", accountNumber, acct_id)
        accountId = acct_id

    params: dict = {
        "dateFrom": dateFrom,
        "dateTo": dateTo,
        "fields": "id,date,amount,amountGross,amountCurrency,description,account(id,number,name),voucher(id,number),customer(id,name),supplier(id,name)",
    }
    if accountId:
        params["accountId"] = accountId
    if supplierId:
        params["supplierId"] = supplierId
    if customerId:
        params["customerId"] = customerId
    result = tx.get("ledger/posting", params)
    values = result.get("values", [])
    logger.info("Found %d postings (accountNumber=%s, accountId=%s)", len(values), accountNumber, accountId)
    if not values:
        return {"found": False, "count": 0, "postings": [], "message": f"No postings found. Searched: accountNumber={accountNumber}, accountId={accountId}, dateFrom={dateFrom}, dateTo={dateTo}. Try broader date range or different account."}
    return {"found": True, "count": len(values), "postings": values}


def create_voucher(
    tx: TripletexClient,
    date: str,
    description: str,
    postings: list[dict],
    **kwargs,
) -> dict:
    """Create a ledger voucher with postings.

    postings: list of {accountNumber: int, debitAmount: float, creditAmount: float, description: str}
    Automatically looks up account IDs from account numbers.
    """
    # Clean up postings: remove junk fields the LLM sends
    cleaned_postings = []
    for p in postings:
        # Strip empty strings, zeros, None values from optional fields
        cleaned = {}
        for k, v in p.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            if k in ("customerId", "supplierId") and isinstance(v, (int, float)) and v == 0:
                continue
            cleaned[k] = v

        # Skip postings where both amounts are 0 — LLM placeholder junk
        debit = cleaned.get("debitAmount", 0) or 0
        credit = cleaned.get("creditAmount", 0) or 0
        if debit == 0 and credit == 0:
            logger.info("Skipping posting with zero amounts: acct=%s desc=%s",
                       cleaned.get("accountNumber"), cleaned.get("description", ""))
            continue

        cleaned_postings.append(cleaned)

    if not cleaned_postings:
        return {"_is_error": True, "_error_summary": "All postings have zero amounts. Provide at least one posting with a non-zero debitAmount or creditAmount."}

    postings = cleaned_postings

    # Look up account IDs on demand (avoids slow bulk fetch)
    if "account_map" not in _cache:
        _cache["account_map"] = {}
    acct_map = _cache["account_map"]

    api_postings = []
    for row_idx, p in enumerate(postings, start=1):
        acct_num = p.get("accountNumber")
        acct_id = acct_map.get(acct_num)
        if not acct_id:
            lookup = tx.get("ledger/account", {"number": str(acct_num)})
            vals = lookup.get("values", [])
            if vals:
                acct_id = vals[0]["id"]
                acct_map[acct_num] = acct_id

        if not acct_id:
            # Auto-create the account with a descriptive name from the posting description
            acct_desc = p.get("description", "")
            # Standard Norwegian account name mappings
            auto_names = {
                1209: "Akkumulerte avskrivninger", 1290: "Akkumulerte avskrivninger annet",
                6010: "Avskrivning bygninger", 6020: "Avskrivning maskiner", 6030: "Avskrivning driftsmidler",
                8700: "Skattekostnad", 2920: "Betalbar skatt", 3400: "Purregebyr",
                7790: "Valutadifferanse", 8050: "Renteinntekt", 8150: "Rentekostnad",
            }
            auto_name = auto_names.get(acct_num, acct_desc[:50] or f"Konto {acct_num}")
            logger.info("Account %s not found — auto-creating as '%s'", acct_num, auto_name)
            create_result = create_ledger_account(tx=tx, number=acct_num, name=auto_name)
            new_id = _get_id(create_result)
            if new_id:
                acct_id = new_id
                acct_map[acct_num] = acct_id
                logger.info("Auto-created account %s '%s' (id=%d)", acct_num, auto_name, acct_id)

        if not acct_id:
            # Still failed — suggest alternatives
            range_start = (acct_num // 100) * 100
            range_end = range_start + 99
            all_accts = tx.get("ledger/account", {"count": 5000})
            similar_vals = [a for a in all_accts.get("values", [])
                           if range_start <= (a.get("number") or 0) <= range_end]
            if not similar_vals:
                # Broaden to the same thousand range
                range_start = (acct_num // 1000) * 1000
                range_end = range_start + 999
                similar_vals = [a for a in all_accts.get("values", [])
                               if range_start <= (a.get("number") or 0) <= range_end]
            suggestions = ", ".join(f"{a['number']} ({a['name']})" for a in similar_vals[:8])
            logger.warning("Account %s not found. Nearby: %s", acct_num, suggestions)
            hint = (f"Account {acct_num} does not exist in this Tripletex account. "
                    f"Available nearby: [{suggestions or 'none'}]. "
                    f"If the prompt SPECIFIES this account number, CREATE it first with create_ledger_account(number={acct_num}, name='...'). "
                    f"Only use a different account if the prompt doesn't specify one.")
            return {"_is_error": True, "_error_summary": hint}

        # Check ledger type — some accounts require subledger references
        # Cache account metadata to avoid repeated lookups
        acct_meta_key = f"acct_meta_{acct_num}"
        if acct_meta_key not in _cache:
            acct_detail = tx.get(f"ledger/account/{acct_id}")
            _cache[acct_meta_key] = acct_detail.get("value", {})
        acct_info = _cache[acct_meta_key]
        ledger_type = acct_info.get("ledgerType", "")

        posting: dict = {"account": {"id": acct_id}, "row": row_idx}
        amt = 0.0
        if p.get("debitAmount") is not None and p["debitAmount"] != 0:
            amt = p["debitAmount"]
        elif p.get("creditAmount") is not None and p["creditAmount"] != 0:
            amt = -p["creditAmount"]
        # Tripletex requires amountGross + amountGrossCurrency — 'amount' alone is silently ignored
        posting["amount"] = amt
        posting["amountGross"] = amt
        posting["amountGrossCurrency"] = amt
        posting["amountCurrency"] = amt
        if p.get("description"):
            posting["description"] = p["description"]

        # Pass through string metadata fields on postings
        for str_field in ("invoiceNumber", "termOfPayment"):
            if p.get(str_field):
                posting[str_field] = str(p[str_field])

        # Pass through entity references on postings
        # Only include refs with valid positive IDs — skip 0 or negative to avoid API errors
        for ref_field in ("supplier", "customer", "project", "department", "employee", "product", "vatType"):
            ref_val = p.get(ref_field) or p.get(f"{ref_field}Id")
            if ref_val:
                if isinstance(ref_val, (int, float)) and int(ref_val) > 0:
                    posting[ref_field] = {"id": int(ref_val)}
                elif isinstance(ref_val, dict) and ref_val.get("id") and int(ref_val["id"]) > 0:
                    posting[ref_field] = ref_val

        # Auto-detect required subledger references from ledger type
        # Account 1500 (Kundefordringer) requires customer, 2400 (Leverandørgjeld) requires supplier
        if ledger_type == "CUSTOMER" and "customer" not in posting:
            # Try to find customer from other postings or from the description
            cust_ref = p.get("customer") or p.get("customerId")
            if cust_ref and isinstance(cust_ref, (int, float)) and int(cust_ref) > 0:
                posting["customer"] = {"id": int(cust_ref)}
            else:
                logger.warning("Account %s (ledgerType=CUSTOMER) requires a customer reference. "
                             "Add 'customerId' to this posting.", acct_num)
                return {"_is_error": True,
                        "_error_summary": f"Account {acct_num} requires a customer reference (ledgerType=CUSTOMER). "
                                          f"Add 'customerId' to the posting for account {acct_num}. "
                                          f"Use find_customer first to get the customer ID."}

        if ledger_type == "VENDOR" and "supplier" not in posting:
            sup_ref = p.get("supplier") or p.get("supplierId")
            if sup_ref and isinstance(sup_ref, (int, float)) and int(sup_ref) > 0:
                posting["supplier"] = {"id": int(sup_ref)}
            else:
                logger.warning("Account %s (ledgerType=VENDOR) requires a supplier reference. "
                             "Add 'supplierId' to this posting.", acct_num)
                return {"_is_error": True,
                        "_error_summary": f"Account {acct_num} requires a supplier reference (ledgerType=VENDOR). "
                                          f"Add 'supplierId' to the posting for account {acct_num}. "
                                          f"Use find_entity(entityType='supplier') first to get the supplier ID."}

        # Support free accounting dimensions on postings
        # Accept both our tool names (customDimension1) and API names (freeAccountingDimension1)
        for i in (1, 2, 3):
            dim_val = p.get(f"customDimension{i}") or p.get(f"freeAccountingDimension{i}")
            if dim_val:
                api_key = f"freeAccountingDimension{i}"
                if isinstance(dim_val, int):
                    posting[api_key] = {"id": dim_val}
                elif isinstance(dim_val, str):
                    dim_value_id = _lookup_dimension_value(tx, dim_val)
                    if dim_value_id:
                        posting[api_key] = {"id": dim_value_id}
                    else:
                        logger.warning("Dimension value '%s' not found for dimension %d", dim_val, i)

        api_postings.append(posting)

    # Validate: postings must balance (sum to 0)
    total = sum(p.get("amount", 0) for p in api_postings)
    if abs(total) > 0.01:
        posting_summary = "; ".join(
            f"acct {postings[i].get('accountNumber')}: {api_postings[i].get('amount', 0)}"
            for i in range(len(api_postings))
        )
        return {"_is_error": True,
                "_error_summary": f"Voucher postings do not balance! Sum = {total:.2f} (must be 0). "
                                  f"Postings: [{posting_summary}]. "
                                  f"Fix: ensure total debits = total credits."}

    body = {
        "date": date,
        "description": description,
        "postings": api_postings,
    }

    # Optional voucher metadata (used by supplier invoice flow)
    voucher_type_id = kwargs.get("_voucherTypeId")
    if voucher_type_id:
        body["voucherType"] = {"id": int(voucher_type_id)}
    ext_voucher_num = kwargs.get("_externalVoucherNumber")
    if ext_voucher_num:
        body["externalVoucherNumber"] = str(ext_voucher_num)

    # Log dimension info for debugging
    for idx, ap in enumerate(api_postings):
        for dim_key in ("freeAccountingDimension1", "freeAccountingDimension2", "freeAccountingDimension3"):
            if dim_key in ap:
                logger.info("Voucher posting %d has %s = %s", idx, dim_key, ap[dim_key])

    result = tx.post("ledger/voucher", body)
    if result.get("_is_error"):
        logger.warning("Voucher creation failed: %s", result.get("_error_summary", ""))
    else:
        logger.info("Created voucher: %s", description)
    return result


def create_supplier_invoice(
    tx: TripletexClient,
    supplierId: int,
    invoiceNumber: str,
    invoiceDate: str,
    dueDate: str | None = None,
    amountInclVat: float = 0,
    accountNumber: int | None = None,
    description: str = "",
    vatTypeId: int = 1,
    lines: list[dict] | None = None,
    **kwargs,
) -> dict:
    """Register an incoming supplier invoice.

    Uses POST /incomingInvoice (not ledger/voucher).
    Single line: use accountNumber + amountInclVat
    Multiple lines: use lines=[{accountNumber, amountInclVat, description, vatTypeId}]
    vatTypeId: 1 = 25% inngående (default), 11 = 15%, 12 = 12%, 0 = no VAT
    """
    import datetime
    # Default dueDate to invoiceDate + 14 days if not provided or same as invoiceDate
    if not dueDate or dueDate == invoiceDate:
        try:
            d = datetime.date.fromisoformat(invoiceDate)
            dueDate = (d + datetime.timedelta(days=14)).isoformat()
        except (ValueError, TypeError):
            dueDate = invoiceDate
    # Resolve accounts on demand (avoids slow full account list fetch)
    def _resolve_account(acct_num):
        if "account_map" not in _cache:
            _cache["account_map"] = {}
        acct_id = _cache["account_map"].get(acct_num)
        if not acct_id:
            lookup = tx.get("ledger/account", {"number": str(acct_num)})
            vals = lookup.get("values", [])
            if vals:
                acct_id = vals[0]["id"]
                _cache["account_map"][acct_num] = acct_id
        return acct_id

    # Build order lines
    import uuid
    if lines:
        order_lines = []
        for i, line in enumerate(lines):
            acct_id = _resolve_account(line.get("accountNumber", accountNumber))
            if not acct_id:
                return {"_is_error": True, "_error_summary": f"Account {line.get('accountNumber')} not found"}
            order_lines.append({
                "externalId": str(uuid.uuid4()),
                "accountId": acct_id,
                "amountInclVat": line.get("amountInclVat", line.get("amount", 0)),
                "description": line.get("description", description),
                "vatTypeId": line.get("vatTypeId", vatTypeId),
                "row": i + 1,
            })
    else:
        acct_id = _resolve_account(accountNumber)
        if not acct_id:
            return {"_is_error": True, "_error_summary": f"Account {accountNumber} not found"}
        order_lines = [{
            "externalId": str(uuid.uuid4()),
            "accountId": acct_id,
            "amountInclVat": amountInclVat,
            "description": description,
            "vatTypeId": vatTypeId,
            "row": 1,
        }]

    body = {
        "invoiceHeader": {
            "vendorId": supplierId,
            "invoiceNumber": invoiceNumber,
            "invoiceDate": invoiceDate,
            "dueDate": dueDate,
            "invoiceAmount": amountInclVat,
            "description": description,
            "currencyId": 1,
        },
        "orderLines": order_lines,
    }

    # Try to activate ELECTRONIC_VOUCHERS module (enables incoming invoice functionality)
    if "incoming_invoice_module_activated" not in _cache:
        modules = tx.get("company/salesmodules")
        module_names = {m.get("name") for m in modules.get("values", [])}
        if "ELECTRONIC_VOUCHERS" not in module_names:
            activate = tx.post("company/salesmodules", {"name": "ELECTRONIC_VOUCHERS"})
            if not activate.get("_is_error"):
                logger.info("Activated ELECTRONIC_VOUCHERS module")
        _cache["incoming_invoice_module_activated"] = True

    result = tx.post("incomingInvoice", body, params={"sendTo": "ledger"})
    if not result.get("_is_error"):
        logger.info("Created supplier invoice %s for supplier %d via incomingInvoice API", invoiceNumber, supplierId)
        return result

    err_summary = result.get("_error_summary", "")
    logger.warning("incomingInvoice API failed: %s", err_summary[:200])

    # Fallback to ledger voucher — let Tripletex auto-split VAT by setting vatType on expense postings
    # This creates proper VAT postings with correct vatType references
    if True:
        logger.info("Falling back to ledger voucher for supplier invoice %s", invoiceNumber)

        kid = kwargs.get("kid")
        project_id = kwargs.get("projectId")

        # Build postings: expense lines with vatType (Tripletex auto-creates VAT postings)
        # + credit 2400 for the total amount
        if lines:
            voucher_postings = []
            total_incl = 0
            for line in lines:
                line_incl = line.get("amountInclVat", line.get("amount", 0))
                line_vat_id = line.get("vatTypeId", vatTypeId)
                acct = line.get("accountNumber", accountNumber or 6590)
                total_incl += line_incl
                posting = {
                    "accountNumber": acct,
                    "debitAmount": line_incl,  # gross incl VAT — Tripletex auto-splits
                    "creditAmount": 0,
                    "description": line.get("description", description) or f"Leverandørfaktura {invoiceNumber}",
                    "vatType": line_vat_id,
                    "supplier": {"id": supplierId},
                    "invoiceNumber": invoiceNumber,
                }
                if dueDate:
                    posting["termOfPayment"] = dueDate
                if project_id and int(project_id) > 0:
                    posting["project"] = int(project_id)
                voucher_postings.append(posting)
            voucher_postings.append({
                "accountNumber": 2400,
                "debitAmount": 0,
                "creditAmount": round(total_incl, 2),
                "description": f"Leverandørfaktura {invoiceNumber}",
                "supplier": {"id": supplierId},
                "invoiceNumber": invoiceNumber,
                "termOfPayment": dueDate or "",
            })
        else:
            voucher_postings = [
                {
                    "accountNumber": accountNumber or 6590,
                    "debitAmount": amountInclVat,  # gross incl VAT — Tripletex auto-splits
                    "creditAmount": 0,
                    "description": description or f"Leverandørfaktura {invoiceNumber}",
                    "vatType": vatTypeId,
                    "supplier": {"id": supplierId},
                    "invoiceNumber": invoiceNumber,
                    "termOfPayment": dueDate or "",
                },
                {
                    "accountNumber": 2400,
                    "debitAmount": 0,
                    "creditAmount": amountInclVat,
                    "description": f"Leverandørfaktura {invoiceNumber}",
                    "supplier": {"id": supplierId},
                    "invoiceNumber": invoiceNumber,
                    "termOfPayment": dueDate or "",
                },
            ]
            if project_id and int(project_id) > 0:
                voucher_postings[0]["project"] = int(project_id)

        # Get Leverandørfaktura voucherType ID (cached)
        if "lev_voucher_type_id" not in _cache:
            vt_result = tx.get("ledger/voucherType")
            _cache["lev_voucher_type_id"] = None
            for vt in vt_result.get("values", []):
                if "Leverandør" in (vt.get("name") or ""):
                    _cache["lev_voucher_type_id"] = vt["id"]
                    break

        voucher_result = create_voucher(
            tx=tx,
            date=invoiceDate,
            description=f"Leverandørfaktura {invoiceNumber}: {description}{' KID: ' + kid if kid else ''}".strip(": "),
            postings=voucher_postings,
            _voucherTypeId=_cache.get("lev_voucher_type_id"),
            _externalVoucherNumber=invoiceNumber,
        )
        if not voucher_result.get("_is_error"):
            logger.info("Created supplier invoice as voucher with voucherType=Leverandørfaktura: %s", invoiceNumber)
        return voucher_result

    return result


def run_payroll(
    tx: TripletexClient,
    employeeId: int,
    date: str,
    year: int | None = None,
    month: int | None = None,
    salaryLines: list[dict] | None = None,
    **kwargs,
) -> dict:
    """Run payroll for an employee.

    salaryLines: [{description, amount, count?, rate?, salaryTypeId?}]
    E.g. [{"description": "Månedslønn", "amount": 55350}, {"description": "Bonus", "amount": 14600}]
    """
    import datetime

    if not year or not month:
        d = datetime.date.fromisoformat(date) if date else datetime.date.today()
        year = year or d.year
        month = month or d.month

    # Look up salary types (1 API call, cached)
    if "salary_types" not in _cache:
        salary_types = tx.get("salary/type")
        st_values = salary_types.get("values", [])
        st_by_name = {}
        fastlonn_id = None
        bonus_id = None
        for st in st_values:
            st_name = (st.get("name") or "").lower()
            st_number = st.get("number", "")
            st_by_name[st_name] = st["id"]
            if st_number:
                st_by_name[st_number] = st["id"]
            if st_name == "fastlønn" or st_number == "2000":
                fastlonn_id = st["id"]
            if st_name == "bonus" or st_number == "2002":
                bonus_id = st["id"]
        _cache["salary_types"] = st_by_name
        _cache["fastlonn_id"] = fastlonn_id
        _cache["bonus_id"] = bonus_id
        _cache["default_salary_type_id"] = fastlonn_id or (st_values[0]["id"] if st_values else None)

    st_by_name = _cache["salary_types"]
    fastlonn_id = _cache["fastlonn_id"]
    bonus_id = _cache["bonus_id"]
    default_salary_type_id = _cache["default_salary_type_id"]

    desc_to_type = {
        "månedslønn": fastlonn_id, "fastlønn": fastlonn_id, "grunnlønn": fastlonn_id,
        "base salary": fastlonn_id, "grundgehalt": fastlonn_id, "salaire de base": fastlonn_id,
        "salario base": fastlonn_id,
        "bonus": bonus_id, "engangsbonus": bonus_id, "prime": bonus_id,
        "bonificación": bonus_id, "einmaliger bonus": bonus_id,
    }

    # Build specifications
    specs = []
    if salaryLines:
        for line in salaryLines:
            amount = line.get("amount", 0)
            spec: dict = {
                "employee": {"id": employeeId},
                "amount": amount,
                "description": line.get("description", ""),
                "year": year,
                "month": month,
                "count": line.get("count", 1) or 1,
                "rate": line.get("rate") if line.get("rate") is not None else amount,
            }

            sal_type_id = line.get("salaryTypeId")
            if not sal_type_id:
                # Try to match description to a salary type
                desc_lower = line.get("description", "").lower()
                # First try explicit mapping
                for keyword, type_id in desc_to_type.items():
                    if keyword in desc_lower and type_id:
                        sal_type_id = type_id
                        break
                # Then try matching against actual salary type names
                if not sal_type_id:
                    for st_name, st_id in st_by_name.items():
                        if st_name and st_name in desc_lower:
                            sal_type_id = st_id
                            break
            if not sal_type_id:
                sal_type_id = default_salary_type_id
            if sal_type_id:
                spec["salaryType"] = {"id": sal_type_id}
            specs.append(spec)
    else:
        # No lines specified — create a single salary line
        spec = {
            "employee": {"id": employeeId},
            "amount": kwargs.get("baseSalary", 0),
            "description": "Månedslønn",
            "year": year,
            "month": month,
        }
        if default_salary_type_id:
            spec["salaryType"] = {"id": default_salary_type_id}
        specs.append(spec)

    # Etterskuddslønn (afterpay): salary for month X is paid on the 10th of month X+1
    payroll_date = date
    if not date or date == datetime.date.today().isoformat():
        if month == 12:
            payroll_date = f"{year + 1}-01-10"
        else:
            payroll_date = f"{year}-{month + 1:02d}-10"

    body = {
        "date": payroll_date,
        "year": year,
        "month": month,
        "payslips": [{
            "employee": {"id": employeeId},
            "date": payroll_date,
            "year": year,
            "month": month,
            "specifications": specs,
        }],
    }

    # Check employment first to avoid 422 error (which costs efficiency points)
    def _ensure_division():
        divisions = tx.get("division")
        for d in divisions.get("values", []):
            return d["id"]
        all_emps = tx.get("employee/employment", {"count": 10})
        for emp in all_emps.get("values", []):
            div = emp.get("division")
            if isinstance(div, dict) and div.get("id"):
                return div["id"]
        whoami = tx.get("token/session/>whoAmI")
        company_id = whoami.get("value", {}).get("companyId")
        company_org = ""
        if company_id:
            company_info = tx.get(f"company/{company_id}")
            company_org = company_info.get("value", {}).get("organizationNumber", "")
        sub_org = str(int(company_org) + 1) if company_org.isdigit() else "974760673"
        div_result = tx.post("division", {
            "name": "Hovedkontor",
            "organizationNumber": sub_org,
            "startDate": f"{year}-01-01",
            "municipality": {"id": 301},
        })
        div_id = _get_id(div_result)
        if div_id:
            logger.info("Created division id=%d for payroll", div_id)
        return div_id

    def _ensure_employment_details(employment_id):
        """Ensure employment has details covering the payroll period."""
        details = tx.get("employee/employment/details", {"employmentId": employment_id})
        if not details.get("values", []):
            logger.info("No employment details — creating for employment %d", employment_id)
            tx.post("employee/employment/details", {
                "employment": {"id": employment_id},
                "date": f"{year}-01-01",
                "employmentType": "ORDINARY",
                "employmentForm": "PERMANENT",
                "remunerationType": "MONTHLY_WAGE",
                "percentageOfFullTimeEquivalent": 100.0,
                "workingHoursScheme": "NOT_SHIFT",
            })

    employments = tx.get("employee/employment", {"employeeId": employeeId})
    emp_values = employments.get("values", [])
    if not emp_values:
        # Create employment before trying salary
        logger.info("No employment found — creating before salary transaction")
        emp_data = tx.get(f"employee/{employeeId}")
        emp_obj = emp_data.get("value", {})
        update_fields: dict = {}
        if not emp_obj.get("dateOfBirth"):
            update_fields["dateOfBirth"] = "1990-01-01"
        if not emp_obj.get("employeeNumber"):
            update_fields["employeeNumber"] = str(employeeId)[-4:]
        if update_fields:
            update_fields["id"] = employeeId
            update_fields["version"] = emp_obj.get("version", 0)
            tx.put(f"employee/{employeeId}", update_fields)

        division_id = _ensure_division()
        emp_body: dict = {"employee": {"id": employeeId}, "startDate": f"{year}-01-01"}
        if division_id:
            emp_body["division"] = {"id": division_id}
        emp_result = tx.post("employee/employment", emp_body)
        employment_id = _get_id(emp_result)
        if employment_id:
            _ensure_employment_details(employment_id)
    else:
        # Employment exists — ensure it has division and details for the period
        employment = emp_values[0]
        employment_id = employment.get("id")
        emp_division = employment.get("division")
        if not emp_division or not (isinstance(emp_division, dict) and emp_division.get("id")):
            # Employment has no division — link it to one (required for salary)
            # First ensure employee has dateOfBirth (required for employment PUT)
            emp_data = tx.get(f"employee/{employeeId}")
            emp_obj = emp_data.get("value", {})
            if not emp_obj.get("dateOfBirth"):
                logger.info("Setting dateOfBirth on employee %d (required for employment update)", employeeId)
                tx.put(f"employee/{employeeId}", {
                    "id": employeeId,
                    "version": emp_obj.get("version", 0),
                    "dateOfBirth": "1990-01-01",
                })

            division_id = _ensure_division()
            if division_id and employment_id:
                logger.info("Linking employment %d to division %d", employment_id, division_id)
                # Re-fetch employment to get fresh version after potential employee update
                fresh_emp = tx.get(f"employee/employment/{employment_id}")
                fresh_version = fresh_emp.get("value", {}).get("version", employment.get("version", 0))
                tx.put(f"employee/employment/{employment_id}", {
                    "id": employment_id,
                    "version": fresh_version,
                    "division": {"id": division_id},
                    "employee": {"id": employeeId},
                    "startDate": employment.get("startDate", f"{year}-01-01"),
                })
        if employment_id:
            _ensure_employment_details(employment_id)

    # Ensure salary settings exist (payment day, standard time) — needed on fresh sandboxes
    _ensure_salary_settings(tx, year)

    # Try salary transaction API with generateTaxDeduction=true
    result = tx.post("salary/transaction", body, params={"generateTaxDeduction": True})
    if not result.get("_is_error"):
        logger.info("Salary transaction created for employee %d, %d-%02d", employeeId, year, month)
        return result

    logger.warning("Salary transaction failed: %s", result.get("_error_summary", "")[:200])

    # If salary transaction fails, fall back to manual voucher on 5000/2940
    logger.info("Falling back to manual payroll voucher on 5000/2940")
    postings = []
    total_amount = 0
    if salaryLines:
        for line in salaryLines:
            amt = line.get("amount", 0)
            if amt <= 0:
                continue
            total_amount += amt
            desc = line.get("description", "Lønn")
            postings.append({
                "accountNumber": 5000, "debitAmount": amt, "creditAmount": 0,
                "description": f"{desc} {year}-{month:02d}", "employee": employeeId,
            })
    else:
        total_amount = kwargs.get("baseSalary", 0)
        if total_amount > 0:
            postings.append({
                "accountNumber": 5000, "debitAmount": total_amount, "creditAmount": 0,
                "description": f"Lønn {year}-{month:02d}", "employee": employeeId,
            })

    if total_amount > 0:
        postings.append({
            "accountNumber": 2940, "debitAmount": 0, "creditAmount": total_amount,
            "description": f"Skyldig lønn {year}-{month:02d}", "employee": employeeId,
        })
        voucher_date = date or datetime.date.today().isoformat()
        voucher_result = create_voucher(
            tx=tx,
            date=voucher_date,
            description=f"Lønn - {year}-{month:02d}",
            postings=postings,
        )
        if not voucher_result.get("_is_error"):
            logger.info("Created payroll voucher for employee %d: %d lines, total %.2f",
                        employeeId, len(postings) - 1, total_amount)
        return {"salary_transaction_error": result.get("_error_summary", ""),
                "voucher": voucher_result}

    return result


def _transform_csv_to_tripletex_format(fileContent: str, accountNumber: str = "1920") -> str:
    """Transform any bank CSV format to Tripletex standard format.

    Tripletex requires: Konto;Kontonavn;Inngående saldo;Utgående saldo;Bokført dato;Forklarende tekst;Ut;Inn
    Common input formats:
    - Dato;Forklaring;Inn;Ut;Saldo
    - Date;Description;Credit;Debit;Balance
    - Already in Tripletex format

    Returns the CSV in Tripletex format.
    """
    import re

    lines = fileContent.strip().split("\n")
    if not lines:
        return fileContent

    header = lines[0].strip().lower()

    # Already in Tripletex format?
    if "konto" in header and "kontonavn" in header and "inngående" in header:
        return fileContent

    # Detect delimiter
    delim = ";" if ";" in lines[0] else ","

    # Parse header to detect columns
    headers = [h.strip().lower() for h in lines[0].split(delim)]

    # Map common column names
    col_map = {}
    for i, h in enumerate(headers):
        if h in ("dato", "date", "bokført dato", "bokf¢rt dato", "transaction date", "valutadato"):
            col_map["date"] = i
        elif h in ("forklaring", "description", "tekst", "forklarende tekst", "text", "beskrivelse"):
            col_map["desc"] = i
        elif h in ("inn", "credit", "inn på konto", "innskudd", "kredit"):
            col_map["inn"] = i
        elif h in ("ut", "debit", "ut fra konto", "uttak", "belastning"):
            col_map["ut"] = i
        elif h in ("saldo", "balance", "beholdning"):
            col_map["saldo"] = i

    if "date" not in col_map or "desc" not in col_map:
        logger.warning("CSV format not recognized, headers: %s — passing through as-is", headers)
        return fileContent

    # Build Tripletex format rows
    tx_lines = ["Konto;Kontonavn;Inngående saldo;Utgående saldo;Bokført dato;Forklarende tekst;Ut;Inn"]

    acct_name = "Bankinnskudd"
    prev_saldo = None

    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split(delim)
        if len(cols) < max(col_map.values()) + 1:
            continue

        date_val = cols[col_map["date"]].strip()
        desc_val = cols[col_map["desc"]].strip()

        inn_val = cols[col_map.get("inn", -1)].strip() if "inn" in col_map and col_map["inn"] < len(cols) else ""
        ut_val = cols[col_map.get("ut", -1)].strip() if "ut" in col_map and col_map["ut"] < len(cols) else ""
        saldo_val = cols[col_map.get("saldo", -1)].strip() if "saldo" in col_map and col_map["saldo"] < len(cols) else ""

        # Parse amounts
        def parse_amount(s):
            s = s.strip().replace(" ", "")
            if not s:
                return 0.0
            try:
                return float(s.replace(",", "."))
            except ValueError:
                return 0.0

        inn_amt = parse_amount(inn_val)
        ut_amt = parse_amount(ut_val)
        saldo_amt = parse_amount(saldo_val)

        # Calculate inngående saldo (opening balance for this line)
        if prev_saldo is not None:
            inng_saldo = prev_saldo
        else:
            # First line: calculate from current saldo minus movement
            inng_saldo = saldo_amt - inn_amt + ut_amt

        utg_saldo = saldo_amt if saldo_amt != 0 else inng_saldo + inn_amt - ut_amt
        prev_saldo = utg_saldo

        # Format amounts — empty string if zero
        inn_str = f"{inn_amt:.2f}" if inn_amt > 0 else ""
        ut_str = f"{ut_amt:.2f}" if ut_amt > 0 else ""

        tx_lines.append(f"{accountNumber};{acct_name};{inng_saldo:.2f};{utg_saldo:.2f};{date_val};{desc_val};{ut_str};{inn_str}")

    result = "\n".join(tx_lines)
    logger.info("Transformed CSV: %d data rows to Tripletex format", len(tx_lines) - 1)
    return result


def import_bank_statement(
    tx: TripletexClient,
    fileContent: str,
    filename: str = "bank_statement.csv",
    accountId: int | None = None,
    bankId: int | None = None,
    fileFormat: str | None = None,
    fromDate: str | None = None,
    toDate: str | None = None,
    **kwargs,
) -> dict:
    """Import a bank statement file (CSV/CAMT/MT940).

    fileContent: the raw file content as string — auto-transformed to Tripletex format
    accountId: ledger account ID (e.g. account 1920). If not provided, looks it up.
    fileFormat: auto-detected. TRIPLETEX_CSV is the default for the proxy.
    fromDate/toDate: date range for the statement (YYYY-MM-DD). Auto-detected from CSV if not provided.
    """
    import datetime
    import re

    # Get bank account ID if not provided
    if not accountId:
        result = tx.get("ledger/account", {"number": "1920"})
        accounts = result.get("values", [])
        if accounts:
            accountId = accounts[0]["id"]

    # Get bank ID if not provided — look up from account
    if not bankId and accountId:
        acct_detail = tx.get(f"ledger/account/{accountId}")
        acct = acct_detail.get("value", {})
        bank = acct.get("bank", {})
        if isinstance(bank, dict):
            bankId = bank.get("id")
        if not bankId:
            banks = tx.get("bank")
            bank_values = banks.get("values", [])
            if bank_values:
                bankId = bank_values[0]["id"]

    # Transform CSV to Tripletex standard format
    transformed = _transform_csv_to_tripletex_format(fileContent)

    # Auto-detect date range from the CSV content
    dates_found = re.findall(r'\d{4}-\d{2}-\d{2}', transformed)
    if dates_found:
        if not fromDate:
            fromDate = min(dates_found)
        if not toDate:
            toDate = max(dates_found)
    if not fromDate:
        fromDate = "2026-01-01"
    if not toDate:
        toDate = datetime.date.today().isoformat()

    # Try TRIPLETEX_CSV first (what the proxy expects), then fall back to other formats
    formats_to_try = []
    if fileFormat:
        formats_to_try.append(fileFormat)
    formats_to_try.extend(["TRIPLETEX_CSV", "DNB_CSV", "NORDEA_CSV", "SPAREBANK1_CSV"])
    # Deduplicate while preserving order
    seen = set()
    formats_to_try = [f for f in formats_to_try if not (f in seen or seen.add(f))]

    last_result = None
    for fmt in formats_to_try:
        params = {
            "accountId": accountId,
            "fromDate": fromDate,
            "toDate": toDate,
            "fileFormat": fmt,
        }
        if bankId:
            params["bankId"] = bankId

        logger.info("Trying bank statement import with format %s", fmt)
        url = f"{tx.base_url}/bank/statement/import"
        resp = tx.client.post(
            url,
            auth=tx.auth,
            params=params,
            files={"file": (filename, transformed.encode("utf-8"), "text/csv")},
        )
        try:
            last_result = resp.json()
        except Exception:
            last_result = {"status": resp.status_code, "message": resp.text[:2000]}

        if resp.status_code < 400:
            logger.info("Imported bank statement with format %s: %s", fmt, filename)
            return last_result

        logger.info("Format %s failed: %s", fmt, last_result.get("message", "")[:200])

    # All formats failed
    logger.warning("Bank statement import failed with all formats")
    last_result["_is_error"] = True
    last_result["_error_summary"] = f"ERROR importing bank statement — tried formats: {formats_to_try}"
    return last_result


def create_bank_reconciliation(
    tx: TripletexClient,
    accountId: int,
    **kwargs,
) -> dict:
    """Create a new bank reconciliation for an account."""
    result = tx.post("bank/reconciliation", {"account": {"id": accountId}})
    logger.info("Created bank reconciliation for account %d", accountId)
    return result


def suggest_bank_matches(
    tx: TripletexClient,
    reconciliationId: int,
    **kwargs,
) -> dict:
    """Suggest matches for a bank reconciliation. Uses query params."""
    url = f"{tx.base_url}/bank/reconciliation/match/:suggest"
    params = {"bankReconciliationId": reconciliationId}
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}

    if resp.status_code >= 400:
        logger.warning("Suggest matches failed: %s", result)
    else:
        logger.info("Suggested matches for reconciliation %d", reconciliationId)
    return result


def find_bank_transactions(
    tx: TripletexClient,
    bankStatementId: int | None = None,
    **kwargs,
) -> dict:
    """Find bank statement transactions."""
    params = {}
    if bankStatementId:
        params["bankStatementId"] = bankStatementId
    result = tx.get("bank/statement/transaction", params)
    values = result.get("values", [])
    logger.info("Found %d bank transactions", len(values))
    if not values:
        return {"found": False, "count": 0, "transactions": [], "message": "No bank transactions found. Has the bank statement been imported?"}
    return {"found": True, "count": len(values), "transactions": values}


def invoice_order(
    tx: TripletexClient,
    orderId: int,
    invoiceDate: str,
    sendToCustomer: bool = False,
    **kwargs,
) -> dict:
    """Convert an existing order into an invoice. Uses PUT /order/{id}/:invoice."""
    _ensure_bank_account(tx)
    params: dict = {"invoiceDate": invoiceDate}
    if sendToCustomer:
        params["sendToCustomer"] = True
    url = f"{tx.base_url}/order/{orderId}/:invoice"
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}
    if resp.status_code >= 400:
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} invoicing order {orderId}"
        logger.warning("Invoice order failed: %s", result.get("_error_summary"))
    else:
        logger.info("Invoiced order %d", orderId)
    return result


def create_project_activity(
    tx: TripletexClient,
    name: str,
    projectId: int | None = None,
    isChargeable: bool = True,
    **kwargs,
) -> dict:
    """Create an activity, optionally linked to a project."""
    activity_type = "PROJECT_GENERAL_ACTIVITY" if projectId else "GENERAL_ACTIVITY"
    body: dict = {"name": name, "isChargeable": isChargeable, "activityType": activity_type}
    result = tx.post("activity", body)
    if result.get("_is_error"):
        logger.warning("Create activity failed: %s", result.get("_error_summary", ""))
        return result
    act_id = _get_id(result)
    logger.info("Created activity '%s' (id=%s)", name, act_id)

    # Link to project if specified
    if projectId and act_id:
        pa_body = {"project": {"id": projectId}, "activity": {"id": act_id}}
        pa_result = tx.post("project/projectActivity", pa_body)
        if pa_result.get("_is_error"):
            logger.warning("Link activity to project failed: %s", pa_result.get("_error_summary", ""))
        else:
            logger.info("Linked activity %s to project %d", act_id, projectId)
    return result


def create_travel_expense_vouchers(
    tx: TripletexClient,
    travelExpenseId: int,
    date: str | None = None,
    **kwargs,
) -> dict:
    """Create vouchers from a travel expense (book it in the ledger)."""
    import datetime
    if not date:
        date = datetime.date.today().isoformat()
    url = f"{tx.base_url}/travelExpense/:createVouchers"
    params = {"id": travelExpenseId, "date": date}
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}
    if resp.status_code >= 400:
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} creating vouchers for travel expense {travelExpenseId}"
    else:
        logger.info("Created vouchers for travel expense %d", travelExpenseId)
    return result


def pay_supplier_invoice(
    tx: TripletexClient,
    invoiceId: int,
    paymentType: int = 0,
    amount: float | None = None,
    paymentDate: str | None = None,
    **kwargs,
) -> dict:
    """Register payment on a supplier invoice. paymentType=0 auto-selects."""
    import datetime
    params: dict = {"paymentType": paymentType}
    if amount is not None:
        params["amount"] = amount
    if paymentDate:
        params["paymentDate"] = paymentDate
    else:
        params["paymentDate"] = datetime.date.today().isoformat()
    url = f"{tx.base_url}/supplierInvoice/{invoiceId}/:addPayment"
    resp = tx.client.post(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}
    if resp.status_code >= 400:
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} paying supplier invoice {invoiceId}"
    else:
        logger.info("Paid supplier invoice %d", invoiceId)
    return result


def approve_supplier_invoice(
    tx: TripletexClient,
    invoiceId: int,
    comment: str | None = None,
    **kwargs,
) -> dict:
    """Approve a supplier invoice."""
    params: dict = {}
    if comment:
        params["comment"] = comment
    url = f"{tx.base_url}/supplierInvoice/{invoiceId}/:approve"
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}
    if resp.status_code >= 400:
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} approving supplier invoice {invoiceId}"
    else:
        logger.info("Approved supplier invoice %d", invoiceId)
    return result


def create_opening_balance(
    tx: TripletexClient,
    date: str,
    postings: list[dict] | None = None,
    **kwargs,
) -> dict:
    """Create an opening balance. postings: [{accountNumber, debitAmount, creditAmount}]"""
    if "account_map" not in _cache:
        _cache["account_map"] = {}
    acct_map = _cache["account_map"]

    api_postings = []
    if postings:
        for row_idx, p in enumerate(postings, start=1):
            acct_num = p.get("accountNumber")
            acct_id = acct_map.get(acct_num)
            if not acct_id:
                lookup = tx.get("ledger/account", {"number": str(acct_num)})
                vals = lookup.get("values", [])
                if vals:
                    acct_id = vals[0]["id"]
                    acct_map[acct_num] = acct_id
            if not acct_id:
                logger.warning("Account %s not found, skipping posting", acct_num)
                continue
            posting: dict = {"account": {"id": acct_id}, "row": row_idx}
            amt = 0.0
            if p.get("debitAmount") is not None and p["debitAmount"] != 0:
                amt = p["debitAmount"]
            elif p.get("creditAmount") is not None and p["creditAmount"] != 0:
                amt = -p["creditAmount"]
            posting["amount"] = amt
            posting["amountGross"] = amt
            posting["amountGrossCurrency"] = amt
            posting["amountCurrency"] = amt

            # Pass through entity references
            for ref_field in ("supplier", "customer", "project", "department", "employee"):
                ref_val = p.get(ref_field) or p.get(f"{ref_field}Id")
                if ref_val:
                    if isinstance(ref_val, (int, float)):
                        posting[ref_field] = {"id": int(ref_val)}
                    elif isinstance(ref_val, dict):
                        posting[ref_field] = ref_val

            api_postings.append(posting)

    body = {"date": date, "postings": api_postings}
    result = tx.post("ledger/voucher/openingBalance", body)
    if result.get("_is_error"):
        logger.warning("Opening balance failed: %s", result.get("_error_summary", ""))
    else:
        logger.info("Created opening balance on %s", date)
    return result


def create_ledger_account(
    tx: TripletexClient,
    number: int,
    name: str,
    **kwargs,
) -> dict:
    """Create a new ledger account."""
    body: dict = {"number": number, "name": name}
    _add_extra_fields(body, kwargs, "Account")
    result = tx.post("ledger/account", body)
    if result.get("_is_error"):
        logger.warning("Create account failed: %s", result.get("_error_summary", ""))
    else:
        logger.info("Created account %d: %s", number, name)
    return result


def create_accounting_dimension(
    tx: TripletexClient,
    dimensionName: str,
    description: str | None = None,
    values: list[str] | None = None,
    **kwargs,
) -> dict:
    """Create a free accounting dimension with optional values.

    values: list of value names, e.g. ["Utvikling", "Internt"]
    Returns the dimension with its values.
    """
    body: dict = {"dimensionName": dimensionName, "active": True}
    if description:
        body["description"] = description

    result = tx.post("ledger/accountingDimensionName", body)
    if result.get("_is_error"):
        logger.warning("Create dimension failed: %s", result.get("_error_summary", ""))
        return result

    dim_id = _get_id(result)
    logger.info("Created accounting dimension '%s' (id=%s)", dimensionName, dim_id)

    # Find the dimensionIndex from the created dimension
    dim_index = result.get("value", result).get("dimensionIndex")
    if dim_index is None:
        # Fallback: look it up
        dims = tx.get("ledger/accountingDimensionName")
        for d in dims.get("values", []):
            if d.get("id") == dim_id:
                dim_index = d.get("dimensionIndex")
                break
        if dim_index is None:
            dim_index = 1  # last resort

    # Create values
    created_values = []
    if values:
        for i, val_name in enumerate(values):
            val_body = {
                "number": val_name,
                "displayName": val_name,
                "dimensionIndex": dim_index,
                "active": True,
                "showInVoucherRegistration": True,
                "position": i + 1,
            }
            val_result = tx.post("ledger/accountingDimensionValue", val_body)
            if val_result.get("_is_error"):
                logger.warning("Create dimension value '%s' failed: %s", val_name, val_result.get("_error_summary", ""))
            else:
                val_id = _get_id(val_result)
                created_values.append({"name": val_name, "id": val_id})
                logger.info("Created dimension value '%s' (id=%s)", val_name, val_id)
                # Update session cache so same-session voucher creation finds these
                if val_id:
                    _cache.setdefault("dimension_values_session", {})[val_name.strip().lower()] = val_id

    result["created_values"] = created_values
    return result


def send_invoice(
    tx: TripletexClient,
    invoiceId: int,
    sendType: str = "EMAIL",
    overrideEmailAddress: str | None = None,
    **kwargs,
) -> dict:
    """Send an invoice to the customer. sendType: EMAIL, EHF, EFAKTURA, LETTER, MANUAL."""
    params: dict = {"sendType": sendType}
    if overrideEmailAddress:
        params["overrideEmailAddress"] = overrideEmailAddress
    url = f"{tx.base_url}/invoice/{invoiceId}/:send"
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}
    if resp.status_code >= 400:
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} sending invoice {invoiceId}"
        logger.warning("Send invoice failed: %s", result.get("_error_summary"))
    else:
        logger.info("Sent invoice %d via %s", invoiceId, sendType)
    return result


def approve_travel_expense(
    tx: TripletexClient,
    travelExpenseId: int,
    **kwargs,
) -> dict:
    """Approve a travel expense."""
    url = f"{tx.base_url}/travelExpense/:approve"
    params = {"id": travelExpenseId}
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}
    if resp.status_code >= 400:
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} approving travel expense {travelExpenseId}"
    else:
        logger.info("Approved travel expense %d", travelExpenseId)
    return result


def deliver_travel_expense(
    tx: TripletexClient,
    travelExpenseId: int,
    **kwargs,
) -> dict:
    """Deliver a travel expense for approval."""
    url = f"{tx.base_url}/travelExpense/:deliver"
    params = {"id": travelExpenseId}
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}
    if resp.status_code >= 400:
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} delivering travel expense {travelExpenseId}"
    else:
        logger.info("Delivered travel expense %d", travelExpenseId)
    return result


def reverse_voucher(
    tx: TripletexClient,
    voucherId: int,
    date: str | None = None,
    **kwargs,
) -> dict:
    """Reverse a voucher (creates a reversing entry)."""
    import datetime
    if not date:
        date = datetime.date.today().isoformat()
    url = f"{tx.base_url}/ledger/voucher/{voucherId}/:reverse"
    params = {"date": date}
    resp = tx.client.put(url, auth=tx.auth, params=params)
    try:
        result = resp.json()
    except Exception:
        result = {"status": resp.status_code, "message": resp.text[:500]}
    if resp.status_code >= 400:
        result["_is_error"] = True
        result["_error_summary"] = f"ERROR {resp.status_code} reversing voucher {voucherId}"
    else:
        logger.info("Reversed voucher %d", voucherId)
    return result


def add_project_participant(
    tx: TripletexClient,
    projectId: int,
    employeeId: int,
    **kwargs,
) -> dict:
    """Add an employee as participant to a project."""
    body = {
        "project": {"id": projectId},
        "employee": {"id": employeeId},
    }
    result = tx.post("project/participant", body)
    if result.get("_is_error"):
        logger.warning("Add participant failed: %s", result.get("_error_summary", ""))
    else:
        logger.info("Added employee %d to project %d", employeeId, projectId)
    return result


def match_bank_transactions(
    tx: TripletexClient,
    reconciliationId: int,
    **kwargs,
) -> dict:
    """Accept suggested matches and match remaining transactions for a bank reconciliation.

    Flow: after suggest_bank_matches, this accepts the suggestions and matches unmatched
    bank transactions by creating counter-postings.
    """
    # 1. Get the reconciliation to find its bank statement
    recon = tx.get(f"bank/reconciliation/{reconciliationId}")
    recon_val = recon.get("value", {})

    # 2. Try to accept suggested matches via the match endpoint
    # Get existing matches/suggestions
    matches = tx.get("bank/reconciliation/match", {"bankReconciliationId": reconciliationId})
    match_values = matches.get("values", [])

    matched_count = len(match_values)
    logger.info("Found %d existing matches for reconciliation %d", matched_count, reconciliationId)

    # 3. Look for unmatched bank statement transactions
    # Get the bank statement linked to this reconciliation
    account = recon_val.get("account", {})
    account_id = account.get("id") if isinstance(account, dict) else None

    # Try to get unmatched transactions
    unmatched = tx.get("bank/statement/transaction", {
        "bankReconciliationId": reconciliationId,
        "isMatched": False,
    })
    unmatched_vals = unmatched.get("values", [])

    if not unmatched_vals:
        # Try without isMatched filter
        all_trans = tx.get("bank/statement/transaction", {
            "bankReconciliationId": reconciliationId,
        })
        unmatched_vals = all_trans.get("values", [])

    logger.info("Found %d bank transactions for reconciliation %d", len(unmatched_vals), reconciliationId)

    # 4. For each unmatched transaction, try to create a match
    new_matches = 0
    for txn in unmatched_vals:
        txn_id = txn.get("id")
        amount = txn.get("amount", 0)
        if txn_id and amount != 0:
            match_body = {
                "bankReconciliation": {"id": reconciliationId},
                "bankStatement": {"id": txn_id},
            }
            match_result = tx.post("bank/reconciliation/match", match_body)
            if not match_result.get("_is_error"):
                new_matches += 1
            else:
                logger.debug("Match failed for txn %d: %s", txn_id, match_result.get("_error_summary", ""))

    return {
        "reconciliationId": reconciliationId,
        "existing_matches": matched_count,
        "new_matches": new_matches,
        "total_transactions": len(unmatched_vals),
    }


def close_bank_reconciliation(
    tx: TripletexClient,
    reconciliationId: int,
    **kwargs,
) -> dict:
    """Close/finalize a bank reconciliation.

    Tries multiple approaches: :close action endpoint, then PUT with isClosed=true.
    """
    # Try 1: :close action endpoint
    url = f"{tx.base_url}/bank/reconciliation/{reconciliationId}/:close"
    resp = tx.client.put(url, auth=tx.auth, params={})
    if resp.status_code < 400:
        try:
            result = resp.json()
        except Exception:
            result = {"status": "closed"}
        logger.info("Closed bank reconciliation %d via :close", reconciliationId)
        return result

    # Try 2: PUT with isClosed=true
    recon = tx.get(f"bank/reconciliation/{reconciliationId}")
    recon_val = recon.get("value", {})
    recon_val["isClosed"] = True
    result = tx.put(f"bank/reconciliation/{reconciliationId}", recon_val)

    if not result.get("_is_error"):
        logger.info("Closed bank reconciliation %d via PUT", reconciliationId)
    else:
        logger.warning("Failed to close reconciliation %d: %s", reconciliationId, result.get("_error_summary", ""))

    return result


def get_account_balances(
    tx: TripletexClient,
    dateFrom: str = "2020-01-01",
    dateTo: str = "2030-12-31",
    accountNumberFrom: int | None = None,
    accountNumberTo: int | None = None,
    **kwargs,
) -> dict:
    """Get aggregated account balances from ledger postings.

    Returns per-account sums of debit/credit amounts. Useful for:
    - Error correction: spotting wrong balances
    - Year-end closing: getting balances to close
    - Verification: checking that postings are correct

    accountNumberFrom/To: filter to a range (e.g. 3000-7999 for revenue+expenses)
    """
    params: dict = {
        "dateFrom": dateFrom,
        "dateTo": dateTo,
        "count": 10000,
        "fields": "id,date,amount,amountGross,amountCurrency,description,account(id,number,name)",
    }

    postings = tx.get("ledger/posting", params)
    values = postings.get("values", [])

    # Aggregate by account
    balances: dict[int, dict] = {}
    for p in values:
        acct = p.get("account", {})
        acct_num = acct.get("number", 0)
        acct_name = acct.get("name", "")
        acct_id = acct.get("id", 0)

        if accountNumberFrom and acct_num < accountNumberFrom:
            continue
        if accountNumberTo and acct_num > accountNumberTo:
            continue

        if acct_num not in balances:
            balances[acct_num] = {
                "accountNumber": acct_num,
                "accountName": acct_name,
                "accountId": acct_id,
                "totalDebit": 0.0,
                "totalCredit": 0.0,
            }

        # Use amountGross (more reliable than amount on some endpoints)
        amount = p.get("amountGross") or p.get("amount") or 0
        if amount > 0:
            balances[acct_num]["totalDebit"] += amount
        elif amount < 0:
            balances[acct_num]["totalCredit"] += abs(amount)

    # Compute net balance and sort by account number
    result_list = []
    for acct_num in sorted(balances.keys()):
        b = balances[acct_num]
        b["balance"] = round(b["totalDebit"] - b["totalCredit"], 2)
        b["totalDebit"] = round(b["totalDebit"], 2)
        b["totalCredit"] = round(b["totalCredit"], 2)
        if b["balance"] != 0:  # Only include accounts with activity
            result_list.append(b)

    logger.info("Got balances for %d accounts (range %s-%s)", len(result_list), accountNumberFrom, accountNumberTo)
    return {"balances": result_list, "count": len(result_list)}


def year_end_closing(
    tx: TripletexClient,
    date: str,
    fiscalYearStart: str | None = None,
    equityAccountNumber: int = 8800,
    **kwargs,
) -> dict:
    """Perform year-end closing: zero out revenue and expense accounts, transfer net to equity.

    Deterministically handles the full year-end closing workflow:
    1. Finds all postings for the fiscal year
    2. Aggregates balances for revenue (3xxx) and expense (4xxx-7xxx) accounts
    3. Creates closing entries that zero them out
    4. Transfers net profit/loss to equity account (default 8800)

    date: closing date (YYYY-MM-DD), typically Dec 31
    fiscalYearStart: start of fiscal year (default Jan 1 of same year)
    equityAccountNumber: equity account for net transfer (default 8800)
    """
    import datetime

    # Determine fiscal year
    closing_date = datetime.date.fromisoformat(date)
    if not fiscalYearStart:
        fiscalYearStart = f"{closing_date.year}-01-01"

    # Get all account balances for the fiscal year
    bal_result = get_account_balances(
        tx=tx, dateFrom=fiscalYearStart, dateTo=date,
        accountNumberFrom=3000, accountNumberTo=8999
    )
    balances = bal_result.get("balances", [])

    if not balances:
        return {"message": "No revenue/expense postings found for the fiscal year", "postings": []}

    # Build closing postings
    closing_postings = []
    total_net = 0.0

    for b in balances:
        acct_num = b["accountNumber"]
        balance = b["balance"]

        # Skip the equity account itself and accounts with no balance
        if acct_num == equityAccountNumber or balance == 0:
            continue

        # Only close revenue (3xxx) and expense (4xxx-7xxx) accounts
        # Revenue accounts (3xxx) typically have CREDIT balances (negative amount in Tripletex)
        # Expense accounts (4xxx-7xxx) typically have DEBIT balances (positive amount)
        if not (3000 <= acct_num <= 7999):
            continue

        # Reversing entry: if balance is positive (debit), credit it; if negative (credit), debit it
        if balance > 0:
            closing_postings.append({
                "accountNumber": acct_num,
                "debitAmount": 0,
                "creditAmount": balance,
                "description": f"Årsoppgjør - {b['accountName']}",
            })
        else:
            closing_postings.append({
                "accountNumber": acct_num,
                "debitAmount": abs(balance),
                "creditAmount": 0,
                "description": f"Årsoppgjør - {b['accountName']}",
            })

        total_net += balance  # Accumulate to transfer to equity

    if not closing_postings:
        return {"message": "No accounts need closing", "postings": []}

    # Add the balancing equity posting
    # If total_net is positive (more debits than credits = net expense), credit equity
    # If total_net is negative (more credits than debits = net revenue), debit equity
    if total_net > 0:
        closing_postings.append({
            "accountNumber": equityAccountNumber,
            "debitAmount": 0,
            "creditAmount": round(total_net, 2),
            "description": f"Årsoppgjør - Årsresultat (underskudd)",
        })
    elif total_net < 0:
        closing_postings.append({
            "accountNumber": equityAccountNumber,
            "debitAmount": round(abs(total_net), 2),
            "creditAmount": 0,
            "description": f"Årsoppgjør - Årsresultat (overskudd)",
        })

    logger.info("Year-end closing: %d accounts, net %.2f to account %d",
                len(closing_postings) - 1, total_net, equityAccountNumber)

    # Create the closing voucher
    result = create_voucher(
        tx=tx,
        date=date,
        description=f"Årsoppgjør {closing_date.year}",
        postings=closing_postings,
    )

    if not result.get("_is_error"):
        logger.info("Year-end closing completed for %d", closing_date.year)
    return result


def search_api(tx: TripletexClient, query: str, **kwargs) -> dict:
    """Search the Tripletex API spec for endpoints matching a query. NO API calls made — reads local OpenAPI spec.

    Use this to discover what API endpoints exist for any operation.
    Returns matching endpoint paths, methods, and summaries.
    """
    from api_reference import search_endpoints
    results = search_endpoints(query)
    if not results:
        return {"found": False, "message": f"No API endpoints matching '{query}'. Try broader terms like 'invoice', 'employee', 'payment', 'voucher', 'salary', 'bank'."}
    logger.info("search_api(%s): %d results", query, len(results))
    return {"found": True, "count": len(results), "endpoints": results}


def get_api_endpoint(tx: TripletexClient, path: str, method: str | None = None, **kwargs) -> dict:
    """Get full documentation for a specific API endpoint. NO API calls made — reads local OpenAPI spec.

    Returns query parameters, request body schema with ALL fields, types, and descriptions.
    Use this to understand exactly what an endpoint accepts before calling generic_post/generic_put.
    """
    from api_reference import get_endpoint_info
    result = get_endpoint_info(path, method)
    logger.info("get_api_endpoint(%s, %s): %s", path, method, "found" if "error" not in result else "not found")
    return result


def get_tool_fields(tx: TripletexClient, entityType: str, **kwargs) -> dict:
    """Return all available fields for a Tripletex entity type from the OpenAPI schema.

    This lets the LLM see every field it can set before calling create/update actions.
    """
    from api_reference import get_schema

    schema = get_schema(entityType)
    if "error" in schema:
        # Try common aliases
        aliases = {
            "employee": "Employee", "customer": "Customer", "product": "Product",
            "invoice": "Invoice", "order": "Order", "orderline": "OrderLine",
            "orderLine": "OrderLine", "project": "Project", "department": "Department",
            "supplier": "Supplier", "contact": "Contact", "travelexpense": "TravelExpense",
            "travelExpense": "TravelExpense", "voucher": "Voucher", "posting": "Posting",
            "salary": "SalaryTransaction", "salaryTransaction": "SalaryTransaction",
            "address": "Address", "travel": "TravelExpense", "cost": "Cost",
            "perdiem": "PerDiemCompensation", "mileage": "MileageAllowance",
        }
        resolved = aliases.get(entityType, aliases.get(entityType.lower()))
        if resolved:
            schema = get_schema(resolved)

    if "error" in schema:
        return {"error": f"Unknown entity type '{entityType}'. Available: Employee, Customer, Product, Invoice, Order, OrderLine, Project, Department, Supplier, Contact, TravelExpense, Voucher, Posting, Address"}

    fields = schema.get("fields", {})
    # Build a concise field list
    result = {}
    for fname, finfo in sorted(fields.items()):
        ftype = finfo.get("type", "string")
        desc = finfo.get("description", "")[:150]
        ref = finfo.get("ref", "")
        enum = finfo.get("enum")

        if "object -> use" in ftype:
            entry = f"ID ({ref})"
        elif enum:
            entry = f"{ftype} enum: {enum}"
        else:
            entry = ftype

        if desc:
            entry += f" — {desc}"
        result[fname] = entry

    # Add custom fields for specific entity types
    entity_lower = schema.get("entity", entityType).lower()
    if entity_lower == "posting":
        result["accountNumber"] = "integer — Account number (e.g. 1920). Auto-resolved to ID."
        result["debitAmount"] = "number — Debit amount (positive)"
        result["creditAmount"] = "number — Credit amount (positive)"
        result["customerId"] = "integer — REQUIRED for account 1500 (Kundefordringer, ledgerType=CUSTOMER)"
        result["supplierId"] = "integer — REQUIRED for account 2400 (Leverandørgjeld, ledgerType=VENDOR)"
    elif entity_lower == "orderline":
        result["unitPrice"] = "number — Alias for unitPriceExcludingVatCurrency"
        result["productNumber"] = "string — Product number to look up (auto-resolved to product ID)"
        result["vatPercent"] = "number — VAT percentage (25, 15, 12, 0), auto-converted to vatTypeId"

    logger.info("get_tool_fields(%s): %d fields", entityType, len(result))
    return {"entityType": schema.get("entity", entityType), "fields": result, "count": len(result)}


def generic_get(
    tx: TripletexClient,
    path: str,
    params: dict | None = None,
    **kwargs,
) -> dict:
    """Generic GET for any endpoint. Fallback for actions not covered above."""
    # Auto-fix common issues
    if params is None:
        params = {}

    # Invoice always needs date params
    if "invoice" in path and "Date" not in str(params):
        params.setdefault("invoiceDateFrom", "2020-01-01")
        params.setdefault("invoiceDateTo", "2030-12-31")

    # Strip query params from path if accidentally included
    if "?" in path:
        parts = path.split("?", 1)
        path = parts[0]
        for pair in parts[1].split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k] = v

    return tx.get(path, params)


def generic_post(
    tx: TripletexClient,
    path: str,
    body: dict,
    **kwargs,
) -> dict:
    """Generic POST fallback. Use specific actions when possible."""
    return tx.post(path, body)


def generic_put(
    tx: TripletexClient,
    path: str,
    body: dict,
    **kwargs,
) -> dict:
    """Generic PUT fallback. Auto-detects :action endpoints and uses query params."""
    # :action endpoints need query params, not body
    if "/:" in path:
        url = f"{tx.base_url}/{path.lstrip('/')}"
        resp = tx.client.put(url, auth=tx.auth, params=body)
        try:
            result = resp.json()
        except Exception:
            result = {"status": resp.status_code, "message": resp.text[:500]}
        if resp.status_code >= 400:
            result["_is_error"] = True
            result["_error_summary"] = f"ERROR {resp.status_code} on PUT {path}"
            logger.warning("generic_put :action failed: %s", result)
        return result
    return tx.put(path, body)


def generic_delete(
    tx: TripletexClient,
    path: str,
    **kwargs,
) -> dict:
    """Generic DELETE fallback."""
    return tx.delete(path)
