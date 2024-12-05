This file describe information in each table. Each table will start with a table name as title and a line of description below then the remaining are columns with its meaning.
# ShipMethod:
Shipping company lookup table.
- ShipMethodID (Primary Key, int, not null): Primary key for ShipMethod records.
- Name (nvarchar(50), not null): Shipping company name.
- ShipBase (money, not null): Minimum shipping charge.
- ShipRate (money, not null): Shipping charge per pound.
- rowguid (uniqueidentifier, not null): ROWGUIDCOL number uniquely identifying the record. Used to support a merge replication sample.
- ModifiedDate (datetime, not null): Date and time the record was last updated.

# Vendor: 
Companies from whom Adventure Works Cycles purchases parts or other goods.
- BusinessEntityID (PK, FK, int, not null): Primary key for Vendor records.  Foreign key to BusinessEntity.BusinessEntityID
- AccountNumber (AccountNnumber(nvarchar(15)), not null): Vendor account (identification) number.
- Name (Name(nvarchar(50)), not null): Company name.
- CreditRating (tinyint, not null): 1 = Superior, 2 = Excellent, 3 = Above average, 4 = Average, 5 = Below average.
- PreferredVendorStatus(Flag(bit), not null): 0 = Do not use if another vendor is available. 1 = Preferred over other vendors supplying the same product.
- ActiveFlag(Flag(bit), not null): 0 = Vendor no longer used. 1 = Vendor is actively used.
- PurchasingWebServiceURL(nvarchar(1024), null): Vendor URL.
- ModifiedDate (datetime, not null): Date and time the record was last updated.

# ProductVendor:
Cross-reference table mapping vendors with the products they supply.
- ProductID (PK, FK, int, not null): Primary key. Foreign key to Product.ProductID.
- BusinessEntityID (PK, FK, int, not null): Primary key. Foreign key to Vendor.BusinessEntityID.
- AverageLeadTime (int, not null): The average span of time (in days) between placing an order with the vendor and receiving the purchased product.
- StandardPrice (money, not null): The vendor's usual selling price.
- LastReceiptCost (money, null): The selling price when last purchased.
- LastReceiptDate (datetime, null): Date the product was last received by the vendor.
MinOrderQty (int, not null): The maximum quantity that should be ordered.
- MaxOrderQty (int, not null): The minimum quantity that should be ordered.
- OnOrderQty (int, not null): The quantity currently on order.
- UnitMeasureCode (FK, nchar(3), not null): The product's unit of measure.
- ModifiedDate (datetime, not null): Date and time the record was last updated.

# PurchaseOrderHeader:
General purchase order information. See PurchaseOrderDetail.
- PurchaseOrderID (PK, int, not null): Primary key. 
- BusinessEntityID (PK, FK, int, not null): Primary key. Foreign key to Vendor.BusinessEntityID.
- RevisionNumber (tinyint, not null): Incremental number to track changes to the purchase order over time.
- Status (tinyint, not null): Order current status. 1 = Pending; 2 = Approved; 3 = Rejected; 4 = Complete.
EmployeeID (FK, int, not null): Employee who created the purchase order. Foreign key to Employee.BusinessEntityID.
- VendorID (FK, int, not null): Vendor with whom the purchase order is placed. Foreign key to Vendor.BusinessEntityID.
- ShipMethodID (FK, int, not null): Shipping method. Foreign key to ShipMethod.ShipMethodID.
- OrderDate (datetime, not null): Purchase order creation date.
- ShipDate (datetime, null): Estimated shipment date from the vendor.
- SubTotal (money, not null): Purchase order subtotal. Computed as SUM(PurchaseOrderDetail.LineTotal)for the appropriate PurchaseOrderID.
- TaxAmt (money, not null): Tax amount.
- Freight (money, not null): Shipping cost.
- TotalDue(Computed, money, not null): Total due to vendor. Computed as Subtotal + TaxAmt + Freight.
- ModifiedDate (datetime, not null): Date and time the record was last updated.

# PurchaseOrderDetail:
Individual products associated with a specific purchase order. See PurchaseOrderHeader.
- PurchaseOrderID (PK, FK, int, not null): Primary key. Foreign key to PurchaseOrderHeader.PurchaseOrderID.
- PurchaseOrderDetailID (PK, int, not null): Primary key. One line number per purchased product.
- DueDate (datetime, not null): Date the product is expected to be received.
- OrderQty (smallint, not null): Quantity ordered.
- ProductID (FK, int, not null): Product identification number. Foreign key to Product.ProductID.
- UnitPrice (money, not null): Vendor's selling price of a single product.
- LineTotal (Computed, money, not null): Per product subtotal. Computed as OrderQty * UnitPrice.
- ReceivedQty (decimal(8,2), not null): Quantity actually received from the vendor.
- RejectedQty (decimal(8,2), not null): Quantity rejected during inspection.
- StockedQty (Computed, decimal(9,2), not null): Quantity accepted into inventory. Computed as ReceivedQty - RejectedQty.
- ModifiedDate (datetime, not null): Date and time the record was last updated.

# Product:
- ProductID (PK, int, not null): Primary key for Product records.
- Name (Name(nvarchar(50)), not null): Name of the product.
- ProductNumber (nvarchar(25), not null): Unique product identification number.
- MakeFlag (Flag(bit), not null): 0 = Product is purchased, 1 = Product is manufactured in-house.
- FinishedGoodsFlag (Flag(bit), not null): 0 = Product is not a salable item. 1 = Product is salable.
- Color (nvarchar(15), null): Product color.
- SafetyStockLevel (smallint, not null): Minimum inventory quantity. 
- ReorderPoint (smallint, not null): Inventory level that triggers a purchase order or work order. 
- StandardCost (money, not null): Standard cost of the product.
- ListPrice (money, not null): Selling price.
- Size(nvarchar(5), null): Product size.
- SizeUnitMeasureCode (FK, nchar(3), null): Unit of measure for Size column.
- WeightUnitMeasureCode (FK, nchar(3), null): Unit of measure for Weight column.
- Weight (decimal(8,2), null): Product weight.
- DaysToManufacture (int, not null): Number of days required to manufacture the product.
- ProductLine (nchar(2), null): R = Road, M = Mountain, T = Touring, S = Standard.
- Class (nchar(2), null): H = High, M = Medium, L = Low.
- Style (nchar(2), null): W = Womens, M = Mens, U = Universal.
- ProductSubcategoryID (FK, int, null): Product is a member of this product subcategory. Foreign key to ProductSubCategory.ProductSubCategoryID.
- ProductModelID (FK, int, null): Product is a member of this product model. Foreign key to ProductModel.ProductModelID.
- SellStartDate (datetime, not null): Date the product was available for sale.
- SellEndDate (datetime, null): Date the product was no longer available for sale.
- DiscontinuedDate (datetime, null): Date the product was discontinued.
- rowguid (uniqueidentifier, not null): ROWGUIDCOL number uniquely identifying the record. Used to support a merge replication sample.
- ModifiedDate (datetime, not null): Date and time the record was last updated.

# SalesOrderDetails:
- SalesOrderID (PK, FK, int, not null): Primary key. Foreign key to SalesOrderHeader.SalesOrderID.
- SalesOrderDetailID (PK, int, not null): Primary key. One incremental unique number per product sold.
- OrderQty (smallint, not null): Quantity ordered per product.
- ProductID(FK, int not null): Product sold to customer. Foreign key to Product.ProductID.
- ModifiedDate (datetime, not null): Date and time the record was last updated.

# BillOfMaterials:
- BillOfMaterialsID (PK, int, not null): Primary key for BillOfMaterials records.
- ProductAssemblyID (FK, int, null): Parent product identification number. Foreign key to Product.ProductID.
- ComponentID (FK, int, not null): Component identification number. Foreign key to Product.ProductID.
- UnitMeasureCode (FK, nchar(3), not null): Standard code identifying the unit of measure for the quantity.
- Bin(tinyint, not null):Storage container on a shelf in an inventory location.
- BOMLevel(smallint,  not null): Indicates the depth the component is from its parent (AssemblyID).
- ModifiedDate (datetime, not null): Date and time the record was last updated.

# ProductInventory:
- ProductID (PK, FK, int, not null): Product identification number. Foreign key to Product.ProductID.
- LocationID (PK, FK, smallint, not null): Inventory location identification number. Foreign key to Location.LocationID.
- Shelf(nvarchar(25), not null): Storage compartment within an inventory location.
- Bin(tinyint, not null):Storage container on a shelf in an inventory location.
- Quantity(smallint, not null): Quantity of products in the inventory location.
- rowguid (uniqueidentifier, not null): ROWGUIDCOL number uniquely identifying the record. Used to support a merge replication sample.
- ModifiedDate (datetime, not null): Date and time the record was last updated.

# ProductSubcategory:
- ProductSubcategoryID (PK, FK, int, not null): Primary key for ProductSubcategory records.
- ProductCategoryID (FK, smallint, not null): Product category identification number. Foreign key to ProductCategory.ProductCategoryID.
- Name(nvarchar(50), not null): Subcategory description.

# ProductCategory:
- ProductCategoryID (PK, FK, int, not null): Primary key for ProductCategory records..
- Name(nvarchar(50), not null): Category description.