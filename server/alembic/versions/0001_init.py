"""
Initial schema: track_points table

Revision ID: 0001_init
Revises: 
Create Date: 2025-09-04 00:00:00
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0001_init'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'track_points',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('channel', sa.String(length=64), nullable=False),
        sa.Column('callsign', sa.String(length=64), nullable=False),
        sa.Column('ts', sa.Float(), nullable=False),
        sa.Column('lat', sa.Float(), nullable=False),
        sa.Column('lon', sa.Float(), nullable=False),
        sa.Column('alt_m', sa.Float(), nullable=True),
        sa.Column('spd_kt', sa.Float(), nullable=True),
        sa.Column('vsi_ms', sa.Float(), nullable=True),
        sa.Column('hdg_deg', sa.Float(), nullable=True),
        sa.Column('pitch_deg', sa.Float(), nullable=True),
        sa.Column('roll_deg', sa.Float(), nullable=True),
        sa.Column('break_path', sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index('idx_track_channel', 'track_points', ['channel'])
    op.create_index('idx_track_callsign', 'track_points', ['callsign'])
    op.create_index('idx_track_channel_callsign_ts', 'track_points', ['channel', 'callsign', 'ts'])


def downgrade() -> None:
    op.drop_index('idx_track_channel_callsign_ts', table_name='track_points')
    op.drop_index('idx_track_callsign', table_name='track_points')
    op.drop_index('idx_track_channel', table_name='track_points')
    op.drop_table('track_points')
