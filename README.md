# Family Identification System

## Overview

This project is a family identification system that helps users discover potential relatives by analyzing their name and photo. The system searches through a structured database of family information, facial embeddings, and relationship data to find matches or potential connections.

## How It Works

1. **User Input**:
   
   - Full name (first and last name)
   - A photo of themselves

2. **Processing Flow**:
   
   - The system checks if the surname exists in the database
   - If found, it locates embeddings of photos from that family
   - If the first name matches a family member, it verifies if it's the same person
   - If only the surname matches, it checks if the person could belong to that family
   - If no surname match, it finds the closest surname matches and checks for potential relatives

## Database Structure

The system uses the following structured database:

- **FIW_PIDs.csv**: Photo lookup table
  
  - PID: Photo ID
  - Name: Surname.firstName
  - URL: Photo URL
  - Metadata: Text caption

- **FIW_FIDs.csv**: Family ID lookup
  
  - FID: Unique family ID
  - Surname: Family name

- **FIW_RIDs.csv**: Relationship types (keys 1-9)

- **FIDs/**: Family directories (FID001-FID1000)
  
  - MID#/: Face images of family members where # is the ID for the person in family file
  - F###*.csv: Family information files containing:
    - Relationship matrix
    - Names
    - Genders

## Example Family Data

A sample family file (FID0001.csv) contains:

| 0   | 1   | 2   | 3   | Name  | Gender |
| --- | --- | --- | --- | ----- | ------ |
| 1   | 0   | 4   | 5   | name1 | female |
| 2   | 1   | 0   | 1   | name2 | female |
| 3   | 5   | 4   | 0   | name3 | male   |

Where the numbers represent relationship types between family members.

## Usage

1. Enter your full name when prompted
2. Upload your photo
3. The system will analyze and display:
   - Exact family matches (if any)
   - Potential relatives in matching families
   - Closest family matches if no exact surname match

## Output

The program will provide one of these results:

1. Exact match confirmation
2. Potential family membership
3. Possible relatives in closest families
