
drop database signature_verification;

CREATE DATABASE signature_verification;

USE signature_verification;

-- Table to store bank employee credentials
CREATE TABLE credentials (
    employee_id CHAR(11) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    employee_email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    branch_id CHAR(7) NOT NULL
);

-- Table to store customer details
CREATE TABLE customer_details (
    account_number CHAR(12) PRIMARY KEY,
    account_holder VARCHAR(255) NOT NULL,
    account_holder_email VARCHAR(255) UNIQUE NOT NULL,
    account_holder_address TEXT NOT NULL,
    signature_image VARCHAR(255)
);

drop table transactions;
drop table product_links;

commit;