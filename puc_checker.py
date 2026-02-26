import json
import os
from datetime import datetime
from typing import Dict, Optional

class PUCChecker:
    """
    Verifies PUC (Pollution Under Control) certificate validity.
    """
    
    def __init__(self, database_path: str):
        """
        Initialize PUC checker with database file.
        
        Args:
            database_path: Path to PUC database JSON file
        """
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"Database not found: {database_path}")
        
        self.database_path = database_path
        self.vehicles = self._load_database()
        print(f"PUC Database loaded with {len(self.vehicles)} vehicles")
    
    def _load_database(self) -> Dict[str, Dict]:
        """
        Load PUC database from JSON file.
        
        Returns:
            Dictionary with vehicle numbers as keys
        """
        with open(self.database_path, 'r') as f:
            data = json.load(f)
        
        # Create dictionary with vehicle number as key for faster lookup
        vehicles = {}
        for vehicle in data.get('vehicles', []):
            vehicle_num = vehicle['vehicle_number'].upper()
            vehicles[vehicle_num] = vehicle
        
        return vehicles
    
    def check_puc_status(self, vehicle_number: str, grace_period_days: int = 0) -> Dict:
        """
        Check PUC status for a vehicle.
        
        Args:
            vehicle_number: Vehicle registration number
            grace_period_days: Grace period in days (default: 0)
            
        Returns:
            Dictionary containing:
                - 'found': Boolean indicating if vehicle is in database
                - 'status': 'Valid', 'Expired', or 'Not Found'
                - 'vehicle_data': Full vehicle information if found
                - 'expiry_date': PUC expiry date
                - 'days_remaining': Days until/since expiration
                - 'in_grace_period': Whether in grace period
        """
        # Normalize vehicle number
        vehicle_num = vehicle_number.upper().replace(" ", "").replace("-", "")
        
        result = {
            'vehicle_number': vehicle_num,
            'found': False,
            'status': 'Not Found',
            'vehicle_data': None,
            'expiry_date': None,
            'days_remaining': None,
            'in_grace_period': False,
            'message': ''
        }
        
        # Check if vehicle exists in database
        if vehicle_num not in self.vehicles:
            result['message'] = f"Vehicle {vehicle_num} not found in PUC database"
            return result
        
        vehicle_data = self.vehicles[vehicle_num]
        result['found'] = True
        result['vehicle_data'] = vehicle_data
        
        # Parse expiry date
        try:
            expiry_date = datetime.strptime(
                vehicle_data['puc_expiry_date'],
                '%Y-%m-%d'
            ).date()
        except ValueError:
            result['status'] = 'Invalid Data'
            result['message'] = "Invalid expiry date format in database"
            return result
        
        result['expiry_date'] = expiry_date.isoformat()
        
        # Calculate days remaining
        current_date = datetime.now().date()
        days_remaining = (expiry_date - current_date).days
        result['days_remaining'] = days_remaining
        
        # Determine status
        if days_remaining >= 0:
            result['status'] = 'Valid'
            result['message'] = f"PUC valid. Expires in {days_remaining} days"
        else:
            # Check if within grace period
            if abs(days_remaining) <= grace_period_days:
                result['status'] = 'Grace Period'
                result['in_grace_period'] = True
                result['message'] = (
                    f"PUC expired {abs(days_remaining)} days ago. "
                    f"In grace period ({grace_period_days} days)"
                )
            else:
                result['status'] = 'Expired'
                result['message'] = f"PUC expired {abs(days_remaining)} days ago"
        
        return result
    
    def search_vehicle(self, vehicle_number: str) -> Optional[Dict]:
        """
        Search for a vehicle in the database.
        
        Args:
            vehicle_number: Vehicle registration number
            
        Returns:
            Vehicle dictionary if found, None otherwise
        """
        vehicle_num = vehicle_number.upper().replace(" ", "").replace("-", "")
        return self.vehicles.get(vehicle_num, None)
    
    def get_all_vehicles(self) -> Dict[str, Dict]:
        """
        Get all vehicles in the database.
        
        Returns:
            Dictionary of all vehicles
        """
        return self.vehicles
    
    def add_vehicle(self, vehicle_number: str, owner_name: str,
                    puc_expiry_date: str, owner_contact: str) -> bool:
        """
        Add a new vehicle to the database.
        
        Args:
            vehicle_number: Vehicle registration number
            owner_name: Owner's name
            puc_expiry_date: PUC expiry date (YYYY-MM-DD)
            owner_contact: Owner's contact information
            
        Returns:
            True if successful, False otherwise
        """
        vehicle_num = vehicle_number.upper().replace(" ", "").replace("-", "")
        
        # Validate date format
        try:
            datetime.strptime(puc_expiry_date, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format: {puc_expiry_date}. Use YYYY-MM-DD")
            return False
        
        # Check if vehicle already exists
        if vehicle_num in self.vehicles:
            print(f"Vehicle {vehicle_num} already exists in database")
            return False
        
        # Add vehicle
        self.vehicles[vehicle_num] = {
            'vehicle_number': vehicle_num,
            'owner_name': owner_name,
            'puc_expiry_date': puc_expiry_date,
            'owner_contact': owner_contact,
            'status': self._determine_status(puc_expiry_date)
        }
        
        # Save to file
        return self._save_database()
    
    def update_vehicle(self, vehicle_number: str, puc_expiry_date: str) -> bool:
        """
        Update PUC expiry date for a vehicle.
        
        Args:
            vehicle_number: Vehicle registration number
            puc_expiry_date: New PUC expiry date (YYYY-MM-DD)
            
        Returns:
            True if successful, False otherwise
        """
        vehicle_num = vehicle_number.upper().replace(" ", "").replace("-", "")
        
        if vehicle_num not in self.vehicles:
            print(f"Vehicle {vehicle_num} not found")
            return False
        
        # Validate date format
        try:
            datetime.strptime(puc_expiry_date, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format: {puc_expiry_date}. Use YYYY-MM-DD")
            return False
        
        # Update vehicle
        self.vehicles[vehicle_num]['puc_expiry_date'] = puc_expiry_date
        self.vehicles[vehicle_num]['status'] = self._determine_status(puc_expiry_date)
        
        # Save to file
        return self._save_database()
    
    def _determine_status(self, puc_expiry_date: str) -> str:
        """
        Determine PUC status based on expiry date.
        
        Args:
            puc_expiry_date: PUC expiry date (YYYY-MM-DD)
            
        Returns:
            Status string ('Valid' or 'Expired')
        """
        expiry_date = datetime.strptime(puc_expiry_date, '%Y-%m-%d').date()
        current_date = datetime.now().date()
        return 'Valid' if (expiry_date - current_date).days >= 0 else 'Expired'
    
    def _save_database(self) -> bool:
        """
        Save database to JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert dictionary back to list format
            vehicles_list = list(self.vehicles.values())
            data = {'vehicles': vehicles_list}
            
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Database saved successfully")
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def print_vehicle_info(self, vehicle_number: str) -> None:
        """
        Print formatted vehicle information.
        
        Args:
            vehicle_number: Vehicle registration number
        """
        result = self.check_puc_status(vehicle_number)
        
        print("\n" + "="*60)
        print(f"PUC CHECK RESULT")
        print("="*60)
        print(f"Vehicle Number: {result['vehicle_number']}")
        print(f"Status: {result['status']}")
        
        if result['found']:
            vehicle = result['vehicle_data']
            print(f"Owner Name: {vehicle['owner_name']}")
            print(f"PUC Expiry Date: {result['expiry_date']}")
            print(f"Days Remaining: {result['days_remaining']}")
            print(f"In Grace Period: {'Yes' if result['in_grace_period'] else 'No'}")
            print(f"Owner Contact: {vehicle['owner_contact']}")
        
        print(f"Message: {result['message']}")
        print("="*60 + "\n")
