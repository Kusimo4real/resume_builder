CREATE TABLE hrms_dashb_appraisal_data (
    log_id VARCHAR(200) PRIMARY KEY,
    account_id VARCHAR(200),
    fullname VARCHAR(200),
    kpi_score VARCHAR(200),
    grade VARCHAR(200),
    qualification_status VARCHAR(200),
    month_date VARCHAR(200),
    department VARCHAR(200),
    domains VARCHAR(200),
    tenant VARCHAR(200)
);

CREATE TABLE hrms_dashb_appraisals (
    appraisal_id VARCHAR(200) PRIMARY KEY,
    employee_id VARCHAR(200),
    manager_id VARCHAR(200),
    appraisal_date VARCHAR(200),
    appraisal_comment VARCHAR(200),
    last_updated VARCHAR(200),
    score_value INTEGER,
    letter_grade VARCHAR(200)
);

CREATE TABLE hrms_dashb_awards (
    num_id VARCHAR(200) PRIMARY KEY,
    award_name VARCHAR(200),
    award_date VARCHAR(200),
    employee_id VARCHAR(200),
    employee_image VARCHAR(512000),
    award_category VARCHAR(200),
    employee_department VARCHAR(200),
    employee_description VARCHAR(200)
);

CREATE TABLE hrms_dashb_baseline (
    log_id VARCHAR(200) PRIMARY KEY,
    dept VARCHAR(200),
    tenant VARCHAR(200),
    baseline INTEGER
);

CREATE TABLE hrms_dashb_department (
    departments VARCHAR(200) PRIMARY KEY
);

CREATE TABLE hrms_dashb_departments (
    dept_name VARCHAR(200) PRIMARY KEY,
    manager_id VARCHAR(200),
    department VARCHAR(200)
);

CREATE TABLE hrms_dashb_employees (
    employee_id VARCHAR(200) PRIMARY KEY,
    subdept_name VARCHAR(200),
    first_name VARCHAR(200),
    last_name VARCHAR(200),
    employee_position VARCHAR(200),
    hire_date VARCHAR(200),
    employee_level INTEGER,
    skills VARCHAR(200),
    project VARCHAR(200),
    salary VARCHAR(200),
    vendor VARCHAR(200),
    location VARCHAR(200),
    external_id VARCHAR(200),
    fullname VARCHAR(300),
    project_two VARCHAR(200),
    department_two VARCHAR(200),
    po_position VARCHAR(200),
    kpi_position VARCHAR(200),
    emp_domain VARCHAR(200),
    work_type VARCHAR(200),
    phone_no VARCHAR(200),
    date_of_birth VARCHAR(200),
    age VARCHAR(200),
    marital_status VARCHAR(200),
    gender VARCHAR(200),
    supervisor VARCHAR(200),
    remark_one VARCHAR(300),
    fo_bo_type VARCHAR(200),
    position_level VARCHAR(200),
    employee_type VARCHAR(200),
    remark_two VARCHAR(300),
    po_source VARCHAR(200),
    work_location VARCHAR(200),
    floor_option VARCHAR(200),
    svc_dept VARCHAR(200),
    c_and_q_level VARCHAR(200),
    email VARCHAR(200),
    dfocp DATE,
    employee_status VARCHAR(200),
    rst_id VARCHAR(200),
    key_resource VARCHAR(100),
    potential_key_resource VARCHAR(100),
    multivendor VARCHAR(200),
    mv_level VARCHAR(50),
    mv_skill VARCHAR(50),
    mv_domain VARCHAR(200),
    mv_vendor_product VARCHAR(200),
    FOREIGN KEY (subdept_name) REFERENCES hrms_dashb_sub_departments(subdept_name)
);

CREATE TABLE hrms_dashb_level (
    emp_level VARCHAR(50) PRIMARY KEY,
    emp_level_new VARCHAR(200)
);

CREATE TABLE hrms_dashb_po_position (
    po_position VARCHAR(200) PRIMARY KEY,
    po_position_new VARCHAR(200)
);

CREATE TABLE hrms_dashb_projects (
    project_name VARCHAR(200) PRIMARY KEY,
    manager_id VARCHAR(200),
    attachment VARCHAR(100000),
    project VARCHAR(200)
);

CREATE TABLE hrms_dashb_province_info (
    addinfo VARCHAR(250) PRIMARY KEY,
    district VARCHAR(128),
    lon_lat_boundary TEXT,
    lon_lat_centry VARCHAR(256)
);

CREATE TABLE hrms_dashb_roles (
    roles_id VARCHAR(200) PRIMARY KEY,
    roles VARCHAR(200)
);

CREATE TABLE hrms_dashb_skills (
    skill_name VARCHAR(200) PRIMARY KEY,
    employee_id VARCHAR(200)
);

CREATE TABLE hrms_dashb_sub_departments (
    subdept_name VARCHAR(200) PRIMARY KEY,
    dept_name VARCHAR(200),
    name VARCHAR(200),
    FOREIGN KEY (dept_name) REFERENCES hrms_dashb_departments(dept_name)
);

CREATE TABLE hrms_dashb_tenant (
    tenant VARCHAR(200) PRIMARY KEY
);

CREATE TABLE hrms_dashb_vendors (
    vendor_id VARCHAR(200) PRIMARY KEY,
    vendor VARCHAR(200)
);